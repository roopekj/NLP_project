# NLP project


## Instructions

**IMPORTANT:** All files related to this project are available at https://github.com/roopekj/NLP_project

Please, contact sergio.hernandezgutierrez@aalto.fi if the following steps are not clear or sufficient to successfully run the application for grading purposes.
All files that are not crucial for the functioning of the project are located under /misc, such as our planning documents and early prototypes.


To run the visualization application, please follow the following steps:

1. Make sure you have Python with version >= 3.10 installed in your computer; earlier versions of Python3 should also work but they have not been tested.

2. In a terminal emulator (e.g., a terminal using Bash), use `cd` commands to place yourself in the submission's root folder and use this terminal for the next steps; for example:

```bash
cd PATH_TO_PROJECT_ROOT
```

3. Install the required Python libraries by executing the following command:
**NOTE:** It's advisable to create a venv (```python3 -m venv ./venv; ```) or otherwise separate this install from your default python environment (conda etc)

```bash
python3 -m pip install -r requirements.txt
```

4. Navigate to the `app` subdirectory and start the server with the commands below (if a prompt appears asking you to allow incoming connections, accept it):

```bash
cd app
python3 -m gunicorn -w 2 -b :8050 app:server --reload --timeout 9000
```

5. Finally, navigate to either one of the following addresses in your web browser: `localhost:8050` or `http://0.0.0.0:8050`
