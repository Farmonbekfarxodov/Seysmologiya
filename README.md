# Seysmologiya Platform

Seismik ma'lumotlarni tahlil qilish va vizualizatsiya qilish platformasi.

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Farmonbekfarxodov/Seysmologiya.git
cd Seysmologiya
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

#### Option A: Migrate from existing user_info.json
If you have an existing `user_info.json` file:
```bash
python scripts/migrate_to_env.py
```

#### Option B: Manual setup
Copy the example environment file:
```bash
cp .env.example .env
```

Then edit `.env` with your configuration.

### 4. Generate a new SECRET_KEY
```bash
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
```

Copy the output and update `SECRET_KEY` in your `.env` file.

### 5. Run database migrations
```bash
python manage.py migrate
```

### 6. Start the development server
```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000`

## ⚙️ Environment Variables

Create a `.env` file in the project root with the following variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key (generate new one) | `django-insecure-...` |
| `DEBUG` | Debug mode (True/False) | `False` |
| `ALLOWED_HOSTS` | Comma-separated list of allowed hosts | `localhost,127.0.0.1,yourdomain.com` |
| `DB_NAME` | Database name | `seismik` |
| `DB_USER` | Database user | `root` |
| `DB_PASSWORD` | Database password | `your_password` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `3306` |
| `CORS_ALLOWED_ORIGINS` | Comma-separated CORS origins | `http://localhost:8000` |

See `.env.example` for a complete template.

## 🔒 Security Notes

**IMPORTANT:**
- **NEVER** commit the `.env` file to version control
- Generate a new `SECRET_KEY` for production
- Set `DEBUG=False` in production
- Use strong database passwords
- Configure `ALLOWED_HOSTS` properly for production
- Review `CORS_ALLOWED_ORIGINS` settings

## 🛠️ Technology Stack

- **Framework**: Django 5.2.4
- **Database**: MySQL
- **API**: Django REST Framework
- **Authentication**: JWT (djangorestframework-simplejwt)
- **Data Analysis**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Folium, Plotly

## 📦 Main Dependencies

- Django 5.2.4
- djangorestframework 3.16.1
- djangorestframework-simplejwt 5.5.1
- mysqlclient 2.2.7
- python-decouple 3.8
- numpy 2.3.2
- pandas 2.3.1
- matplotlib 3.10.5
- folium 0.20.0

## 🏗️ Project Structure

```
Seysmologiya/
├── seismo_project/          # Main project settings
├── seismos_app/            # Seismic data app
├── download_base_app/      # Data download functionality
├── upload_catalog_app/     # Catalog upload functionality
├── app_users/              # User management
├── app_informativlik/      # Information/statistics app
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates
├── media/                  # User uploaded files
├── scripts/                # Utility scripts
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── manage.py              # Django management script
```

## 🧪 Development

### Running Tests
```bash
python manage.py test
```

### Creating Superuser
```bash
python manage.py createsuperuser
```

### Collecting Static Files
```bash
python manage.py collectstatic
```

## 📝 License

This project is licensed under the terms specified in the repository.

## 👥 Contributors

- Farmonbek Farxodov

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.
