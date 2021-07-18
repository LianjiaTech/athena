# coding=utf-8
# Copyright (C) ATHENA AUTHORS; MingShen; JianweiSun
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support eager mode
# pylint: disable=too-few-public-methods, no-member, too-many-arguments, unused-argument

import sys
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    """
    use native logging module
    support:
        DEBUG INFO WARNING ERROR CRITICAL
    default log level is WARNING

    log name is `paltform_tools`
    """
    level_dict={
        "DEBUG"   : logging.DEBUG,
        "debug"   : logging.DEBUG,
        "INFO"    : logging.INFO,
        "info"    : logging.INFO,
        "WARNING" : logging.WARNING,
        "warning" : logging.WARNING,
        "ERROR"   : logging.ERROR,
        "error"   : logging.ERROR,
        "CRITICAL"   : logging.CRITICAL,
        "critical"   : logging.CRITICAL,
    }
    def __init__(self,logdir=None):
        # this could make log print to console besides log file
        #logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
        #                    datefmt='%Y-%m-%d %I:%M:%S')
        if logdir != None:
            self.logger = logging.getLogger("")
            self.logger.setLevel(logging.INFO)
            self.ch = RotatingFileHandler(logdir, encoding="utf-8", backupCount = 10, maxBytes = 9073741824000000000)
            self.ch.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt="", datefmt='')
            self.ch.setFormatter(formatter)
            self.logger.addHandler(self.ch)
            self.default_ch = self.ch
        else:
            self.logger = logging.getLogger("")
            self.logger.setLevel(logging.INFO)
            self.ch = logging.StreamHandler(stream=sys.stdout)  # output to standard output
            self.ch.setLevel(logging.INFO)
            self.format = logging.Formatter(fmt="", datefmt='')  # output format
            self.ch.setFormatter(self.format)
            self.logger.addHandler(self.ch)
            self.default_ch = self.ch

    def reset(self):
        self.logger.removeHandler(self.ch)
        self.logger.addHandler(self.default_ch)
        self.ch=self.default_ch

    def set_out_file(self, filename):
        ch=RotatingFileHandler(filename, encoding="utf-8", backupCount=10, maxBytes=9073741824000000000000)
        ch.setLevel(logging.INFO)
        formatter=logging.Formatter(fmt="", datefmt='')
        ch.setFormatter(formatter)

        self.logger.removeHandler(self.ch)
        self.logger.addHandler(ch)
        self.ch=ch

    def setLevel(self, level_string):
        if level_string not in Logger.level_dict:
            self.logger.critical("set the wrong log level: {}".format(level_string))
            return False
        else:
            self.logger.setLevel(Logger.level_dict[level_string])

    def debug(self, log_str):
        self.logger.debug(log_str)

    def info(self, log_str):
        self.logger.info(log_str)

    def warning(self, log_str):
        self.logger.warning(log_str)

    def error(self, log_str):
        self.logger.error(log_str)

    def critical(self, log_str):
        self.logger.critical(log_str)

LOG=Logger()#


if __name__ == "__main__":
    pass
