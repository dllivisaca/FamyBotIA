import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, CURRENT_DIR)

from api.app import app
from a2wsgi import ASGIMiddleware

application = ASGIMiddleware(app)
