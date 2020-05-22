# Use the python 3.7.5 container image
FROM python:3.7.5-stretch

# Set the working directory to app
WORKDIR /app

# Copy the current directory contents into container at /app
ADD . /app

# Install dependdencies
RUN pip install -r requirements.txt

# Run the command to start uWSGI
CMD ["uwsgi", "app.ini"]