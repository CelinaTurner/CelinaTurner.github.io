## Retail Analytics on Databricks

**Project description:** An end-to-end lakehouse on the public [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii), built on Databricks with a medallion architecture and transformed with **dbt**. Raw transactions are ingested incrementally with Auto Loader, cleaned and modeled into business-ready marts by dbt (with data-quality tests at every layer), governed by Unity Catalog, orchestrated as code with a Databricks Asset Bundle, and validated in GitHub Actions CI. The goal was a complete, reproducible data platform — ingestion through CI — rather than a single analysis notebook.

[View the repository on GitHub](https://github.com/CelinaTurner/databricks-retail-lakehouse)

<img src="https://raw.githubusercontent.com/CelinaTurner/databricks-retail-lakehouse/main/images/databricks-lakehouse-architecture.svg"/>

### 1. Incremental ingestion (Bronze)

Rather than a one-time file load, raw CSVs land in a Unity Catalog volume and are ingested incrementally by Auto Loader, which tracks schema and only processes new files. The write uses an `availableNow` trigger so the same notebook runs cleanly as a scheduled batch task:

```python
(spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", f"{landing}/_schema")
    .load(landing)
 .writeStream
    .option("checkpointLocation", f"{landing}/_checkpoint")
    .trigger(availableNow=True)
    .toTable("retail_portfolio.bronze.online_retail"))
```

A subtle but instructive issue surfaces right here: the source column `Customer ID` contains a space, which Delta rejects in column names. Sanitizing the names at ingest (so `Customer ID` becomes `Customer_ID`) keeps the physical names portable and ensures Bronze and the downstream dbt models agree on one spelling.

### 2. Transformation and testing with dbt (Silver → Gold)

dbt owns the transformation layers. Silver (`stg_online_retail`) casts types, parses the non-ISO `M/d/yyyy H:mm` timestamps, dedupes, and drops cancellations and non-positive rows. Gold builds three marts: a `fct_sales` fact at invoice-line grain with a surrogate key, a `dim_customers` dimension with RFM-style recency/frequency/monetary measures, and a `revenue_by_country` rollup.

The fact's surrogate key was the instructive one: the source has no line identifier and contains genuine duplicate physical lines, so a hash over the natural columns collided on ~9,000 rows. Adding a `row_number()` line sequence inside the key makes every physical line unique without dropping any revenue.

Every model carries tests, so a build fails loudly if the data drifts:

```yaml
- name: sales_key
  data_tests: [not_null, unique]
- name: customer_id
  data_tests:
    - relationships:
        to: ref('dim_customers')
        field: customer_id
```

A full `dbt build` materializes the silver view and the three gold marts, then runs all 24 tests:

<img src="https://raw.githubusercontent.com/CelinaTurner/databricks-retail-lakehouse/main/images/dbt_build_pass.png"/>

### 3. Governance with Unity Catalog

The medallion layers are schemas (`bronze` / `silver` / `gold`) inside a single Unity Catalog catalog, so lineage, access control, and discovery are consistent across the pipeline rather than notebook-local.

### 4. Orchestration as code

The pipeline is defined as a Databricks Asset Bundle (`databricks.yml`) — a job whose first task runs the Auto Loader ingestion and whose second runs `dbt build`, with the dependency enforced. Deploying it is `databricks bundle deploy`, so the orchestration lives in version control instead of being wired up by hand.

<img src="https://raw.githubusercontent.com/CelinaTurner/databricks-retail-lakehouse/main/images/DAG-Retail_Lakehouse.png"/>

A bundle run executes both tasks end to end on serverless compute — ingestion, then the dbt transform and tests:

<img src="https://raw.githubusercontent.com/CelinaTurner/databricks-retail-lakehouse/main/images/Success-Retail_Lakehouse.png"/>

### 5. CI/CD

A GitHub Actions workflow runs `dbt parse` on every pull request — fast, and it never touches the warehouse, so it costs no compute. The full `dbt build` with tests runs on demand. Connection details are GitHub secrets; nothing sensitive is committed.

<img src="https://raw.githubusercontent.com/CelinaTurner/databricks-retail-lakehouse/main/images/dbt_workflow_run.png"/>

### 6. Takeaways

The dataset is small, but the project demonstrates the full shape of a production lakehouse and spans the data stack: incremental ingestion, dbt transformation and testing, catalog-level governance, orchestration-as-code, and CI. Each piece is a real, runnable artifact in the repo rather than a description, which is the point — it shows the platform being built and validated, not just understood.
