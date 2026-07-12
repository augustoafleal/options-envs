# Versioning

Environment-facing changes that can affect learning behavior should be versioned explicitly.

Typical examples:

- Gymnasium environment IDs such as `OptionsEnv/Pinball-v0`
- task names such as `default-v0`
- layout or asset revisions when they change behavior

When in doubt, prefer creating a new versioned task, layout, or environment ID instead of silently changing behavior in place.
