# Logger Module

This is the logger module used by the scripts in `src/` to produce logs in `logs/log.txt`

### Usage of logger module

You can import the logger with `from logger.logger import logger`.
Then you can use the logger as follows:

```
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
```

Logger statements with info level and up are also printed to console and are color coded. All other levels are always
printed to the log file.

### Best practices

When you catch errors, log it as follows:

```
try:
    # some code
except Exception as e:
    logger.error("Error message: {}".format(e))
```

When you log in main use `logger.info`:

```
def main()
    logger.info("Starting main")
```

When you log in a function 1 deep use `logger.info` with 3 --- and a space:

```
def function1()
    logger.info("--- Starting function1")
```

When you log in a function 2 deep, use `logger.debug` (so only to file) with 6 --- and a space:

```
def function2()
    logger.debug("------ Starting function2")
```
