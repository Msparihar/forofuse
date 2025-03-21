# Contributing to Forofuse

We love your input! We want to make contributing to Forofuse as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue that pull request!

## Development Setup

1. Clone your fork of the repository:

```bash
git clone https://github.com/YOUR_USERNAME/forofuse.git
cd forofuse
```

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

3. Start Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. Run tests:

```bash
pytest
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt if you add or remove dependencies
3. The PR will be merged once you have the sign-off from maintainers

## Any contributions you make will be under the MIT License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/YOUR_USERNAME/forofuse/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
