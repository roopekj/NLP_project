# NLP project


## Instructions

Please, contact sergio.hernandezgutierrez@aalto.fi if the following steps are not clear or sufficient to successfully run the application for grading purposes.

To run the visualization application, please follow the following steps:

1. Make sure you have Python with version >= 3.10 installed in your computer; earlier versions of Python3 should also work but they have not been tested.

2. In a terminal emulator (e.g., a terminal using Bash), use `cd` commands to place yourself in the submission's root folder and use this terminal for the next steps; for example:

```bash
cd ~/Downloads/team10_submission
```

3. Install the required Python libraries by executing the following command:

```bash
python3 -m pip install -r requirements.txt
```

4. Navigate to the `app` subdirectory and start the server with the commands below (if a prompt appears asking you to allow incoming connections, accept it):

```bash
cd app
python3 -m gunicorn -w 2 -b :8050 app:server --reload --timeout 9000
```

5. Finally, navigate to either one of the following addresses in your web browser: `localhost:8050` or `http://0.0.0.0:8050`
