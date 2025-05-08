# Deploying Your RAG Assistant Web App

Follow these steps to make your application accessible globally through a public URL.

## Option 1: Deploying to Render.com (Free)

[Render](https://render.com/) offers a free tier for web services that allows you to deploy your application with a public URL.

### Steps:

1. **Sign up for a Render account**
   - Go to [render.com](https://render.com/) and sign up for a free account

2. **Create a new Web Service**
   - Click on "New" and select "Web Service"
   - Connect your GitHub account or use the "Deploy from Existing Repository" option

3. **Configure your service**
   - Name: `rag-assistant` (or any name you prefer)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python web_app.py`
   - Plan: Free

4. **Set Environment Variables**
   - Add the following environment variable:
     - Key: `FLASK_ENV`
     - Value: `production`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

Your application will be available at a URL like `https://rag-assistant.onrender.com` (depending on the name you chose).

## Option 2: Using ngrok (Temporary URL)

If you want to quickly get a temporary public URL for your local application:

1. **Install ngrok**
   - Download from [ngrok.com](https://ngrok.com/) and sign up for a free account

2. **Start your application locally**
   ```
   python web_app.py
   ```

3. **In a new terminal, start ngrok**
   ```
   ngrok http 5000
   ```

4. **Copy the HTTPS URL**
   - ngrok will display a forwarding URL like `https://12345abcde.ngrok.io`
   - This URL is accessible globally as long as ngrok is running

## Option 3: Deploying to Heroku (Paid)

Heroku used to offer a free tier but now requires a credit card:

1. **Add a Procfile**
   Create a file named `Procfile` with:
   ```
   web: waitress-serve --port=$PORT web_app:app
   ```

2. **Deploy to Heroku**
   ```
   heroku create
   git push heroku main
   ```

## Option 4: Using PythonAnywhere (Free with limitations)

1. Sign up at [PythonAnywhere](https://www.pythonanywhere.com/)
2. Upload your project files
3. Create a new web app with Flask
4. Configure it to point to your web_app.py file

## Ready to use

Whichever method you choose, once deployed, your RAG Assistant will be accessible globally through the provided URL. Share this URL with anyone who wants to test your application. 