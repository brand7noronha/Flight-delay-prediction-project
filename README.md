# Flight-delay-prediction-project
This project is a web app using which a user can check the flight delay based on the flight details and can plan his/her flight accordingly.

## External Flight API Keys

To enable lookup by flight number, place your API keys in `config.py` or set the matching environment variables:

- `AVIATIONSTACK_API_KEY`
- `FLIGHTRADAR24_API_KEY`

The application uses Aviationstack as the primary lookup provider and falls back to Flightradar24 when configured.
