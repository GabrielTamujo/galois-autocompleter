from flask_restful import Resource, abort
from flask import jsonify, request, Response
from flask import current_app as app
import json

import gpt_2_simple as gpt2

import logging
logger = logging.getLogger(__name__)

