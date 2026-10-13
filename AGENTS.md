# Project intent

Endless Voices is a hobby project for exploring LLMs in game
conversations. Good engineering should make experimentation easier
and keep the code understandable. Optimize for a small codebase
that one person can comfortably maintain.

- Treat experiments as disposable by default. Keep enough code,
  inputs and notes to understand the result. Do not turn experiments
  into maintained frameworks or give them dedicated test suites.
- Judge complexity across the whole change. Individually reasonable
  helpers, tests, configuration and documentation can collectively
  be excessive. Prefer the smallest complete solution.
- Make scope growth visible while working. If a task starts requiring
  substantially more code or structure than expected, explain why
  and identify the simpler option before continuing that expansion.
- Engineer for present requirements. Do not add speculative
  abstractions, extension points or backwards compatibility.
- Match reporting detail to the change. For substantial changes,
  explain the design choices and maintenance cost, not just that
  implementation and tests are complete.
