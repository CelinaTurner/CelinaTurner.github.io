## Snowflake RBAC as Code

**Project description:** A complete Snowflake account — warehouses, databases, schemas, and a role-based access control model — defined entirely in Terraform and deployed through GitHub Actions. The motivation was to bring the same infrastructure-as-code discipline that engineering teams use for cloud infrastructure to the data platform itself: every warehouse, role, and grant lives in version control, changes go through a pull request with a visible plan, and nothing is clicked together by hand in the Snowflake UI.

[View the repository on GitHub](https://github.com/CelinaTurner/snowflake-rbac-terraform)

<img src="images/snowflake-rbac-architecture.svg?raw=true"/>

### 1. The access-role / functional-role pattern

The core design decision is to split roles into two kinds, which is the pattern Snowflake recommends for any account beyond a small one:

- **Access roles** are bound to a specific object and privilege level — for example, "read everything in `ANALYTICS`" or "write to `RAW`." They are never granted directly to people.
- **Functional roles** map to a job function (loader, transformer, analyst) and *are* granted to users. Each functional role is assembled by inheriting whichever access roles that function needs.

The effective chain is `USER → FUNCTIONAL ROLE → ACCESS ROLE(S) → object privileges`, with all functional roles rolling up to `SYSADMIN`. Onboarding a person becomes "grant them `FR_ANALYST`," and changing what analysts can touch is editing a single inheritance edge rather than hunting down scattered grants.

In Terraform, roles are declared as lists and created with `for_each` so the model stays declarative:

```hcl
locals {
  access_roles     = ["RAW_READ", "RAW_WRITE", "ANALYTICS_READ", "ANALYTICS_WRITE"]
  functional_roles = ["LOADER", "TRANSFORMER", "ANALYST"]
}

resource "snowflake_account_role" "access" {
  for_each = toset(local.access_roles)
  name     = "AR_${each.value}_${var.environment}"
}

resource "snowflake_account_role" "functional" {
  for_each = toset(local.functional_roles)
  name     = "FR_${each.value}_${var.environment}"
}
```

Inheritance is expressed with `snowflake_grant_account_role` — for instance, the transformer reads `RAW` and writes `ANALYTICS`:

```hcl
resource "snowflake_grant_account_role" "transformer_raw_read" {
  role_name        = snowflake_account_role.access["RAW_READ"].name
  parent_role_name = snowflake_account_role.functional["TRANSFORMER"].name
}
```

### 2. Grants on current *and* future objects

Privileges attach only to access roles. The grants cover warehouse usage, database and schema usage, and table/view privileges — and critically, both existing objects and **future** ones, so a newly created table in `ANALYTICS` is readable by analysts without any further Terraform run:

```hcl
resource "snowflake_grant_privileges_to_account_role" "read_tables_future" {
  for_each          = { RAW_READ = "raw", ANALYTICS_READ = "analytics" }
  account_role_name = snowflake_account_role.access[each.key].name
  privileges        = ["SELECT"]

  on_schema_object {
    future {
      object_type_plural = "TABLES"
      in_database        = snowflake_database.this[each.value].name
    }
  }
}
```

This uses the current `snowflakedb/snowflake` v2 provider and its `snowflake_grant_privileges_to_account_role` resource. (Many examples still circulating online use the deprecated `Snowflake-Labs/snowflake` source and the older `snowflake_role` / `snowflake_*_grant` resources — part of the work was migrating the mental model to the v2 API.)

### 3. Cost guardrails

Each workload gets its own warehouse — loading, transforming, reporting — so credit usage is attributable rather than pooled, and each can be sized independently. An account resource monitor caps monthly credits and suspends warehouses at the quota, which is cheap insurance against a runaway query in a demo or dev account:

```hcl
resource "snowflake_resource_monitor" "monthly" {
  name                      = "RM_MONTHLY_${var.environment}"
  credit_quota              = var.monthly_credit_quota
  frequency                 = "MONTHLY"
  notify_triggers           = [80, 90]
  suspend_trigger           = 100
  suspend_immediate_trigger = 110
}
```

### 4. CI/CD with plan-on-PR, apply-on-merge

A GitHub Actions workflow enforces the review gate. On a pull request it runs `fmt -check`, `init`, `validate`, and `plan`, then posts the plan back as a PR comment so a reviewer approves the exact diff. Only a merge to `main` triggers `apply`:

```yaml
- name: Terraform apply
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: terraform apply -input=false -auto-approve tfplan
```

Authentication uses a dedicated service user with key-pair (JWT) auth. Every credential is supplied as a `TF_VAR_*` environment variable locally and as a GitHub Actions secret in CI — no secret is ever committed to the repository.

### 5. Takeaways

The exercise made concrete how much of "DataOps" is really just applying ordinary software-engineering practice to data infrastructure: declarative definitions, code review, automated validation, and least-privilege access expressed as data rather than tribal knowledge. Natural next steps are remote state (the `azurerm` backend is stubbed in), per-environment workspaces, and pulling warehouse sizing from environment-specific variable files.
