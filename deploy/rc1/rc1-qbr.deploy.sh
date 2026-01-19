# Set up deployment folder
rm -rf ./deployment_files
mkdir deployment_files

# Copy common files for all deployments
uv export --format requirements-txt --no-hashes --no-dev --output-file ./deployment_files/requirements.txt
cp -rf ./src ./deployment_files/src
cp app.py ./deployment_files

# Upload files to databricks
cd deployment_files

# Sync
cp ./../deploy/rc1/rc1-qbr.app.yaml app.yaml

databricks sync . /Workspace/Users/damian.beltritti@consultants.lgads.tv/databricks_apps/rc1-qbr-agent-d1

# Back to root
cd ..

# Deploy rc1 version
databricks apps deploy rc1-qbr-agent --source-code-path /Workspace/Users/damian.beltritti@consultants.lgads.tv/databricks_apps/rc1-qbr-agent-d1

# Cleanup deployment files
rm -rf ./deployment_files
