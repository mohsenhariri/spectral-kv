# Python Template

A [simple] [general-purpose] Python template 🐍🚀🎉🦕


I used this template for [fesenjoon](https://pypi.org/project/fesenjoon/) and [medviz](https://pypi.org/project/medviz/).

## How to use

- Linux and Mac
  Use GCC Makefile

- Install Makefile

  https://community.chocolatey.org/packages/make

  http://www.cygwin.com/

### Bootstrap

Set a name for the package:

```bash
    make init newName
```

To use env in pool:

``` bash
    make env
```

There is no need to active virtual environment when the `PY` command is used from Makefile.

To use env here (local directory)

``` bash
    make env-local
```

``` bash
    source env_platform_ver/bin/activate
```
Check Python and pip version

``` bash
    make
```

Update pip and build tools

``` bash
    make check
```

Install the requirements

``` bash
    make pireq
```

### Install a package

``` bash
    make piu numpy matplotlib scipy
```

## Features

- Linter: Pylint
- Formatter: Black
- CI: GitHub Actions

### ToDo

- [x] Formatter: Black + isort
- [x] Type checker: MyPy
- [x] Linter: Ruff
- [x] Linter: Pylint
- [x] GitHub Actions
- [x] Git Hooks
- [x] PyPI Publish
- [x] Flit
- [x] Poetry
- [x] Ruff 

### Git

Git hooks are available in ./scripts/.githooks

``` bash
    chmod +x ./scripts/.githooks/script

    git config core.hooksPath ./scripts/.githooks

```


## Publish to PyPI


1. To build a package, run:
``` bash
    make pkg-build
```

2. To check the build, run:
``` bash
    make pkg-check
```

3. To install the package locally, run: 
``` bash
    make pkg-install
```

4. Create `.pypirc` file in the root directory of the project. It should look like this:

``` bash
    [distutils]
    index-servers =
        pypi
        testpypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: <your username>
    password: <your password>

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: <your username>
    password: <your password>
```

4. To publish to PyPI, run:

``` bash
    make pkg-publish
```
