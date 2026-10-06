import importlib
import os
import sys
from unittest import TestCase
from unittest.mock import patch


class SettingsFallbackTests(TestCase):
    SETTINGS_MODULE = 'seismo_project.settings'

    def _load_settings(self, env):
        with patch.dict(os.environ, env, clear=True):
            sys.modules.pop(self.SETTINGS_MODULE, None)
            return importlib.import_module(self.SETTINGS_MODULE)

    def test_sqlite_is_used_when_db_name_is_missing(self):
        settings = self._load_settings({})
        self.assertEqual(settings.DATABASES['default']['ENGINE'], 'django.db.backends.sqlite3')
        self.assertEqual(settings.CORS_ALLOWED_ORIGINS, [])

    def test_mysql_is_used_when_db_name_is_present(self):
        settings = self._load_settings({
            'DB_NAME': 'seismo',
            'DB_USER': 'user',
            'DB_PASSWORD': 'pass',
            'DB_HOST': '127.0.0.1',
            'DB_PORT': '3307',
        })
        self.assertEqual(settings.DATABASES['default']['ENGINE'], 'django.db.backends.mysql')
        self.assertEqual(settings.DATABASES['default']['NAME'], 'seismo')
