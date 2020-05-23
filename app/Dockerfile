# Use the python container image
# FROM python:3.7.5-stretch
FROM python:3.7-buster

# Set the working directory to app
WORKDIR /app

# Copy the current directory contents into container at /app
ADD . /app

# Install dependdencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the command to start uWSGI
CMD ["uwsgi", "app.ini"]
# CMD ["gunicorn", "-w", "4", "run:app"]