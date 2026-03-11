# ADR-004: Helm + GitOps Deployment

## Status
Accepted — 2026-02-18

## Context
The project started with raw Kubernetes YAML manifests in `deploy/k8s/`. As the deployment
grew (canary, HPA, PDB, NetworkPolicy, PrometheusRule), managing environment-specific
configurations became error-prone:

- Staging vs production differ in replica count, resource limits, and feature flags
- Manual `kubectl apply` is risky and unauditable
- No drift detection if someone manually edits a live resource

## Decision

### Helm Chart (`deploy/helm/hotel-booking-cancellation-prediction/`)
Parameterize all Kubernetes resources via Helm templates:

```
deploy/helm/hotel-booking-cancellation-prediction/
├── Chart.yaml
├── values.yaml           # defaults
├── values-staging.yaml   # staging overrides
├── values-production.yaml # production overrides
├── .helmignore
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── pdb.yaml
    ├── network-policy.yaml
    ├── canary.yaml
    ├── prometheus-rule.yaml
    └── NOTES.txt
```

**Key design choices**:
- Feature flags via values (`canary.enabled`, `autoscaling.enabled`, `monitoring.prometheusRule.enabled`)
- Secrets referenced by name (not embedded in values)
- Environment-specific values files layered on top of defaults

### GitOps (ArgoCD primary, Flux alternative)

**ArgoCD configuration** (`deploy/gitops/argocd/`):
- `project.yaml` — RBAC boundary (source repo + destination namespaces)
- `staging.yaml` — Auto-sync with self-heal, pruning enabled
- `production.yaml` — Manual sync (requires approval in CI or ArgoCD UI)

**Flux configuration** (`deploy/gitops/flux/`):
- `GitRepository` → `HelmRelease` for both environments
- Production uses `suspend: true` until approved

### Deployment Flow
```
git push main
  → CI builds image + pushes to GHCR
  → ArgoCD detects out-of-sync
  → Staging: auto-sync + smoke tests
  → Production: manual approve → sync
```

## Consequences

### Positive
- Single source of truth in Git (auditable, reversible)
- Environment parity with layered values files
- Drift detection + self-healing in staging
- Production requires explicit approval (safety gate)

### Negative
- Helm templating adds complexity vs raw YAML
- ArgoCD/Flux requires cluster-level infrastructure
- Two GitOps tools provided (team must choose one)

### Migration
Original `deploy/k8s/` manifests are preserved for reference but should be
considered deprecated in favor of the Helm chart.
