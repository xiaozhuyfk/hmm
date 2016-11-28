"""
File: globals.py
Author: Hongyu Li (hongyul)
"""

from ConfigParser import SafeConfigParser
import logging

logger = logging.getLogger(__name__)

config = None

def read_configuration(configfile):
    global config
    global alpha, lam, output_dim, iterations, layer_dim

    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser

    return parser
