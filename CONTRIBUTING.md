# Contributing to Awesome Python Robotics

Thanks for your interest! This list aims to be the canonical pointer to high-quality, actively maintained Python projects for robotics.

## Adding an entry

1. Find the right section. If your entry spans multiple, place it where it is most prominent.
2. Add the entry **alphabetically** within the section.
3. Use this format:
   ```markdown
   - [Project Name](https://github.com/org/repo) — One-line description ending with a period.
   ```
4. Link to the canonical project page (usually a GitHub repo, or the project's own docs if the source lives elsewhere).
5. Keep the description **under one line** and focused on what makes the project useful for robotics work in Python.

## Quality bar

Entries should be:

- **Python-first** — wrappers around C++/Julia/MATLAB projects are fine if the Python API is the primary user-facing interface.
- **Actively maintained** — a commit in the last ~18 months, or clearly stable & widely used.
- **Open source** — pinned to a free, redistributable license.
- **Robotics-relevant** — perception, planning, control, simulation, hardware I/O, learning for robots, or a closely related domain.

Pure deep-learning frameworks (e.g., PyTorch) are listed only once in the *Relevant Python Libraries* section, not in every applied subsection that uses them.

## Removing or correcting entries

PRs that remove abandoned projects or fix broken links are very welcome. Please mention in the PR description what made you flag the entry.

## License

By contributing, you agree that your contributions will be released under the [CC0 1.0](LICENSE) public-domain dedication.
