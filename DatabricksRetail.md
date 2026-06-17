## Retail Analytics on Databricks

**Project description:** An end-to-end lakehouse built on Databricks using the public [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — roughly a year of transactions from a UK-based online retailer. The project implements the medallion architecture (Bronze → Silver → Gold), governs the data with Unity Catalog, orchestrates the refresh with a Databricks Workflows DAG, and adds SQL alerting on the business layer. Everything runs on Serverless compute. The aim was to demonstrate the modern Databricks platform features rather than just a single notebook of analysis.

<img src="images/databricks-medallion-architecture.svg?raw=true"/>

### 1. Medallion architecture

The pipeline refines data quality in three progressive layers, each persisted as Delta:

- **Bronze** — raw ingestion of the source file with columns as-landed and full history retained. No business logic, so it's always possible to replay downstream layers from source.
- **Silver** — cleaned and conformed: types cast, duplicates removed, cancelled and zero/negative-quantity rows filtered out, and column names normalized.
- **Gold** — business-ready marts aggregated for consumption: revenue by country, customer RFM segments, and monthly cohort metrics.

A subtle but instructive bug surfaced at the Silver layer. The source data has a column that arrives with awkward formatting, and writing it straight through caused Delta write failures because the column name contained characters Delta rejects. Renaming `customer_id` (and standardizing the other column names) in the Silver transform resolved it — a good reminder that the Silver layer's job includes making names safe for everything downstream:

```python
silver_df = (
    bronze_df
      .withColumnRenamed("Customer ID", "customer_id")
      .dropDuplicates()
      .filter(F.col("quantity") > 0)
      .filter(~F.col("invoice").startswith("C"))   # drop cancellations
      .withColumn("line_revenue", F.col("quantity") * F.col("unit_price"))
)
```

### 2. Unity Catalog governance

The three layers live as schemas inside a single Unity Catalog catalog (`retail_portfolio` → `bronze` / `silver` / `gold`). Centralizing the namespace under Unity Catalog means lineage, access control, and discovery all work consistently across the layers rather than being notebook-local, which is the whole point of governing a lakehouse rather than just running notebooks against files.

### 3. Orchestration with Workflows

A Databricks Workflows job ties the layers into a scheduled DAG with explicit task dependencies — ingest must complete before clean, which must complete before aggregate. Modeling it as a dependency graph (rather than one monolithic notebook) means a failure is isolated to its task and the job is restartable from the point of failure.

### 4. SQL alerting

On top of the Gold marts, Databricks SQL alerts run threshold checks — for example, flagging if daily revenue or order volume falls outside an expected band — and notify when a condition trips. This closes the loop from raw file to monitored business metric without leaving the platform.

### 5. Serverless throughout

All compute — the transformation jobs and the SQL warehouse behind the alerts — runs on Databricks Serverless, so there are no clusters to size or keep warm and cost tracks actual usage. For a portfolio/demo workload that spins up intermittently, this is both simpler and cheaper than managing dedicated clusters.

### 6. Takeaways

The dataset itself is modest, but the point of the project is the *shape* of a production lakehouse: clear separation of raw/refined/business layers, governance applied centrally, orchestration as a dependency graph, and monitoring on the metrics that matter. The medallion pattern generalizes directly to far larger and messier sources, which is what makes it worth practicing on something small and public.
