# Security Policy

## Scope

SphereBrain is experimental research software and is not currently intended for safety-critical, medical, diagnostic, financial, or production decision-making.

## Reporting a vulnerability or exposed secret

Before a private reporting channel is configured, do not open a public issue containing:

- API keys, access tokens, passwords, or credentials;
- personal information;
- a working exploit that could expose another person's system or data.

Instead, use GitHub's private vulnerability reporting feature if it is enabled for this repository. If no private channel is available, report only that a private security contact is required, without including the sensitive details.

A permanent security contact or private reporting route must be configured before the repository is announced publicly.

## Supported versions

Before the first public release, only the current publication candidate is reviewed for disclosure and archive preparation. Historical experimental versions are preserved for research provenance and may not receive security fixes.

## Secrets

Never commit secrets. Use environment variables or local files excluded by `.gitignore`. If a secret is committed:

1. revoke or rotate it immediately;
2. assess where it was exposed;
3. remove it from current files;
4. decide whether Git history must be rewritten before publication;
5. document the remediation without publishing the secret value.

Deleting a file from the latest commit does not remove it from Git history.

## Research safety

Unexpected outputs, corrupted state, high resource use, or irreversible modification of experimental data should be reported with the exact commit and reproduction conditions. Preserve original evidence before attempting a repair.

## Disclaimer

Security review is incomplete while the repository remains in publication preparation. No public release should be interpreted as a guarantee of fitness or security.
