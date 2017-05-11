# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn import log
from abc import ABC, abstractmethod
import argparse
import sys
import traceback
import colorama as col


class App(ABC):
    """A helper class for building scripts/applications.

    The logic of the app should be placed in :func:`run`. To define command
    line arguments, override :func:`define_args` and :func:`test_args`.

    To print inside the app, use either the logger with :func:`warning`,
    :func:`info`, :func:`debug1` and :func:`debug2`, or color print
    with :func:`print1` and :func:`print2`. Finally, one can simply use
    the regular Python :func:`print`. To report a fatal error, use
    :func:`error`. All output (including exceptions) is saved to a file if
    ``--out`` is specified in the command line.

    Args:
        description (str): App description.
    """

    # Logging from within the app
    logger = log.get_logger()
    info = logger.info
    warning = logger.warning
    debug1 = logger.debug1
    debug2 = logger.debug2

    class StreamFork():
        """Forks a stream to another stream and a file."""

        def __init__(self, stream, file):
            self.stream = stream
            self.file = file

        def write(self, message):
            self.stream.write(message)
            self.file.write(message)

        def flush(self):
            self.stream.flush()
            self.file.flush()

    def __init__(self, description):
        self.description = 'LibSPN: ' + description
        col.init()

    def main(self):
        """Main function of the app."""
        # Argument parsing
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.define_args(parser)
        other_params = parser.add_argument_group(title="other")
        other_params.add_argument('-v', '--debug1', action='store_true',
                                  help="print log messages at level DEBUG1")
        other_params.add_argument('-vv', '--debug2', action='store_true',
                                  help="print log messages at level DEBUG2")
        other_params.add_argument('-o', '--out', type=str,
                                  metavar='FILE',
                                  help="save output to FILE")
        self.args = parser.parse_args()
        # Redirect copy of output to a file
        self._orig_stdout = sys.stdout  # Used to not copy color codes to file
        self._orig_stderr = sys.stderr  # Used to not copy color codes to file
        if self.args.out:
            self._out_file = open(self.args.out, 'w')
            sys.stdout = App.StreamFork(sys.stdout, self._out_file)
            sys.stderr = App.StreamFork(sys.stderr, self._out_file)
        else:
            self._out_file = None
        # Configure logger to output to the new stderr at specified level
        if self.args.debug2:
            log_level = log.DEBUG2
        elif self.args.debug1:
            log_level = log.DEBUG1
        else:
            log_level = log.INFO
        log.config_logger(log_level, stream=sys.stderr)
        # Test and print
        self._print_header()
        self.test_args()
        # Run the app
        try:
            self.run()
        except Exception as e:
            # Print exception traceback to save it to file before
            # the file is closed in finally
            print(traceback.format_exc(), end='')
            sys.exit(1)
        finally:
            if self._out_file is not None:
                # Revert streams and close file
                sys.stderr = self._orig_stderr
                sys.stdout = self._orig_stdout
                log.config_logger(log_level, stream=sys.stderr)
                self._out_file.close()

    @abstractmethod
    def run(self):
        """Implement app functionality here."""

    @abstractmethod
    def define_args(self, parser):
        """Define argparse arguments here.

        Args:
            parse (argparse.ArgumentParser): The parser.
        """

    def test_args(self):
        """Test values of arguments in ``self.args`` here."""

    def print1(self, msg):
        """Print with color 1."""
        if self._out_file is not None:
            print(msg, file=self._out_file)
        print(col.Fore.YELLOW + msg + col.Style.RESET_ALL,
              file=self._orig_stdout)

    def print2(self, msg):
        """Print with color 2."""
        if self._out_file is not None:
            print(msg, file=self._out_file)
        print(col.Fore.BLUE + msg + col.Style.RESET_ALL,
              file=self._orig_stdout)

    def error(self, msg=None):
        """Report an error and exit the app."""
        msg = "ERROR: " + str(msg)
        if msg is not None:
            if self._out_file is not None:
                print(msg, file=self._out_file)
            print(col.Fore.RED + msg + col.Style.RESET_ALL,
                  file=self._orig_stderr)
        sys.exit(1)

    def _print_header(self):
        self.print1("======================================")
        self.print1(self.description)
        self.print1("======================================")
        self.print1("Args:")
        for name, val in sorted(vars(self.args).items()):
            if name not in {'out', 'debug1', 'debug2'}:
                self.print1("- %s: %s" % (name, val))
        self.print1("======================================")
