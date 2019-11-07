# Contributing to FLAME GPU 2

Looking to contribute to FLAME GPU 2? **Here's how you can help.**

Please take a moment to review this document in order to make the contribution process easy and effective for everyone involved.

## Reporting Bugs

1 - Use the GitHub issue search -- check if the issue has already been reported.
2 - Check if the issue has been fixed -- try to reproduce it with the latest `master` or development branch.
3 - Isolate the problem -- ideally create a [minimal example](https://stackoverflow.com/help/minimal-reproducible-example).
4 - Complete the [bug report template](https://github.com/FLAMEGPU/FLAMEGPU2_dev/issues/new?template=bug_report.md&title=[BugReport]) -- provide as much relevant detail as possible.

A good bug report should contain all the information necessary to allow a developer to reproduce the issue, without needing to ask for further information.
Please try to be as detailed as possible in your report. What is your environment? What steps will reproduce the issue? What CUDA version, GPU and OS experience the problem? What is the expected outcome?
All of these details assist the process of investigating and fixing bugs.

## Requesting Features

Feature requests are welcome, FLAME GPU 2 is under active development and as a general agent-based modelling framework aims to provide all functionality required by modellers. However, it's up to *you* to make a strong case to convince the project's developers of the merits and priority of this feature. Please provide as much detail and context as possible.

Complete the [feature request template](https://github.com/FLAMEGPU/FLAMEGPU2_dev/issues/new?template=feature_request.md&title=[FeatureReq]).

## Submitting Pull Requests

Good pull requests—patches, improvements, new features—are a fantastic help. They should remain focused in scope and avoid containing unrelated commits.

**Please ask first** before embarking on any significant pull request (e.g. implementing features, refactoring code), otherwise you risk spending a lot of time working on something that the project's developers might not want to merge into the project.

Please adhere to the [coding conventions](#Coding-Conventions) used throughout the project (indentation, accurate comments, etc.) and any other requirements (such as test coverage).

Before merging pull requests are required to pass all continuous integration:

* The full codebase should build with the `WARNINGS_AS_ERRORS` option enabled under the provided CMake configuration, on both Windows and Linux.
* The full test suite should pass.
* `cpplint` should report no issues, using `CPPLINT.cfg` found in the root of the project.

Ideally, these should be addressed as far as possible before creating a pull request. Executing the tests and linter are detailed in the main [README](README.md).

### Unit Tests

Any significant additional feature will require creation of tests to demonstrate their functional correctness.

The existing [tests](tree/master/tests) can be used as templates for how to implement a new test.


### Coding Conventions

A consist code style throughout

The code within FLAME GPU conforms to the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html), as enforced by `cpplint`, with a number of relaxations:

* `[Formatting/Line Length]`: Relaxed to a limit of 240 characters.

<!--### Naming Conventions
??????
--->

#### Checking Code Style

As detailed within the main [README](README.md), if `cpplint` is installed CMake will generate lint targets which report code which strays from the projects coding style.

This can be executed using `make lint_all` or building the `lint_all` project.

*Note: If adding new files, they will first need to be added to `CMakeLists.txt` and CMake configure re-ran.*

### Updating CMakeLists.txt

If adding new source or header files, it is necessary to add them to the list of files within the relevant `CMakeLists.txt`:

The main FLAME GPU 2 library: within [`src`](tree/master/src).
The example models: within the models subdirectory of [`examples`](tree/master/examples).
The tests: within [`tests`](tree/master/tests).

Similarly if submitting a new example model, it will need to include it's own `CMakeLists.txt` (which can be based off an existing examples). This will also need to be reference from the main `CMakeLists.txt` in the root directory of the project.

This ensure that users on both Linux and Windows can continue to build FLAME GPU 2.

## License
By contributing your code, you agree to license your contribution under the [MIT License](LICENSE).