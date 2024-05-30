# Use the official PostgreSQL image from Docker Hub
FROM postgres:14

# Copy the pg_dump file into the container

# Expose PostgreSQL port (default: 5432)
EXPOSE 5432

# Set environment variables from .env file
ENV POSTGRES_DB=""
ENV POSTGRES_USER=""
ENV POSTGRES_PASSWORD=""
ENV POSTGRES_HOST=127.0.0.1

# Run the command to import the database dump
CMD ["bash", "-c", "psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/backup.sql"]
