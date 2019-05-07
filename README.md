# CSE 576

## Installation

1. Install Xcode from the App Store. Then, install Xcode Command Line Tools.
1. Install Open Source QT from https://www.qt.io/download. The minimum installation is fine.
1. Add QT to the path. On macOS, add

    ```
    export PATH="$HOME/Qt/$QT_VERSION/clang_64/bin:$PATH"
    ```

    to your `$HOME/.bash_profile`. Substitute `$QT_VERSION` as necessary.

## Homework 1

Inside the `hw1/Code` directory, launch the application with `qmake` and `make`.

```
qmake ImgFilter.pro [CONFIG+=debug] && make && ./ImgFilter.app/Contents/MacOS/ImgFilter &
```

Run tests with `qmake` and `make check`.

```
qmake Project1Test.pro && make check
```

## Homework 2

Similar to [Homework 1](#homework-1), launch the application with `qmake` and
`make` from the `hw2/code` directory.

```
qmake ImageStitcher.pro CONFIG+=debug && make && ./ImageStitcher.app/Contents/MacOS/ImageStitcher &
```

## Homework 3

Similar to [Homework 1](#homework-1), launch the application with `qmake` and
`make` from the `hw3/Code` directory.

```
qmake ImageRetrieval.pro CONFIG+=debug && make && ./ImageRetrieval.app/Contents/MacOS/ImageRetrieval &
```
