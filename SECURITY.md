# Security Policy

## Reporting a vulnerability

Please report suspected security vulnerabilities privately to the project maintainers rather than opening a public issue that contains exploit details or secrets.

A useful report includes the affected version, the behavior observed, reproduction steps and the potential impact.

## Security principles

Lattice does not log prompts or model output by default, does not configure application logging and excludes raw provider objects from normal serialization. Provider integrations are responsible for redacting provider-specific credentials and secrets.

Applications should avoid placing secrets in prompts, tool descriptions, tool results or provider options unless those values are intentionally required by the selected provider.
