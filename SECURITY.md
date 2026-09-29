# Security policy

## Supported versions

Security fixes are applied to the latest version on the default branch.

## Reporting a vulnerability

Do not disclose credentials, exploit details, or sensitive data in a public issue.
Use GitHub's **Report a vulnerability** form on the repository Security tab when
private vulnerability reporting is available. Otherwise, contact the repository
owner through their GitHub profile to arrange a private report.

Include the affected component, reproduction steps, impact, and any suggested
mitigation. Please allow a reasonable period for investigation and remediation
before public disclosure.

## Deployment boundary

The included Docker Compose configuration is intended for a trusted local machine.
The Streamlit, FastAPI, MCP, and MLflow services do not implement production
authentication. Do not expose them directly to an untrusted network; deploy behind
an authenticated reverse proxy and apply normal network, TLS, and secrets-management
controls.
