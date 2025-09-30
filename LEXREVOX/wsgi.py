"""
WSGI config for LEXREVOX project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangÑ
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'LEXREVOX.settings')

application = get_wsgi_application()

app = application