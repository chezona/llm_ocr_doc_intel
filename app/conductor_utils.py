"""
This module provides a client for interacting with a Netflix Conductor server API.
It allows for registering and starting workflows, and checking their status.
"""
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConductorClient:
    """
    A client for programmatically interacting with the Netflix Conductor workflow engine API.
    Supports operations like checking health, registering workflow and task definitions,
    starting workflows, and retrieving workflow status.
    """
    
    def __init__(self, server_api_url: str = "http://localhost:8080/api"):
        """
        Initializes the Conductor client.
        Args:
            server_api_url: The base API URL of the Conductor server.
        """
        self.server_api_url = server_api_url.rstrip('/') # Ensure no trailing slash
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        logger.info(f"ConductorClient initialized with API URL: {self.server_api_url}")

    def check_connection(self, timeout: int = 5) -> bool:
        """
        Checks if the Conductor server is available and healthy.
        Args:
            timeout: Request timeout in seconds.
        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self.server_api_url}/health", headers=self.headers, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            logger.debug(f"Conductor server health check successful: {response.json()}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Conductor server at {self.server_api_url}/health: {e}")
            return False

    def register_workflow_definition(self, workflow_definition: Dict[str, Any]) -> bool:
        """
        Registers a single workflow definition with Conductor.
        Args:
            workflow_definition: A dictionary representing the workflow definition.
        Returns:
            True if registration was successful (or if workflow might already exist - 200/204/409 typically), False otherwise.
        """
        if not isinstance(workflow_definition, dict):
            logger.error("Workflow definition must be a dictionary.")
            return False
        try:
            url = f"{self.server_api_url}/metadata/workflow"
            logger.debug(f"Registering workflow: {workflow_definition.get('name')} at {url}")
            response = requests.post(url, headers=self.headers, json=workflow_definition)
            
            # Conductor might return 204 for success, or 409 if it already exists and overwrite=false (default)
            if response.status_code in [200, 204]: # 200 can happen if it already existed and no change
                logger.info(f"Successfully registered/updated workflow: {workflow_definition.get('name')}")
                return True
            # Some versions/configurations might return 409 if it exists and an update is not forced.
            # Treating 409 as a success if the goal is just to ensure it's there.
            elif response.status_code == 409:
                logger.info(f"Workflow {workflow_definition.get('name')} already exists (409 Conflict). Registration considered successful.")
                return True
            else:
                logger.error(f"Failed to register workflow {workflow_definition.get('name')}. Status: {response.status_code}, Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error registering workflow {workflow_definition.get('name')}: {e}")
            return False

    def register_task_definitions(self, task_definitions: List[Dict[str, Any]]) -> bool:
        """
        Registers a list of task definitions with Conductor.
        Args:
            task_definitions: A list of dictionaries, each representing a task definition.
        Returns:
            True if all tasks were registered successfully, False otherwise.
        """
        if not isinstance(task_definitions, list) or not all(isinstance(td, dict) for td in task_definitions):
            logger.error("Task definitions must be a list of dictionaries.")
            return False
        if not task_definitions:
            logger.info("No task definitions provided to register.")
            return True # Or False, depending on desired strictness for empty list
        try:
            url = f"{self.server_api_url}/metadata/taskdefs"
            task_names = ", ".join([td.get('name','UNKNOWN_TASK') for td in task_definitions])
            logger.debug(f"Registering task definitions: {task_names} at {url}")
            response = requests.post(url, headers=self.headers, json=task_definitions)
            
            if response.status_code in [200, 204]: # Conductor typically returns 204 for successful task def creation/update
                logger.info(f"Successfully registered/updated task definitions: {task_names}")
                return True
            else:
                logger.error(f"Failed to register task definitions. Status: {response.status_code}, Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error registering task definitions: {e}")
            return False

    def start_workflow(self, workflow_name: str, workflow_input: Dict[str, Any], version: Optional[int] = None, correlation_id: Optional[str] = None) -> Optional[str]:
        """
        Starts a new workflow execution.
        Args:
            workflow_name: The name of the workflow to start.
            workflow_input: A dictionary containing the input data for the workflow.
            version: Optional specific version of the workflow to start.
            correlation_id: Optional correlation ID for the workflow instance.
        Returns:
            The workflow instance ID if successfully started, None otherwise.
        """
        try:
            url = f"{self.server_api_url}/workflow/{workflow_name}"
            payload = {
                "name": workflow_name,
                "version": version,
                "input": workflow_input,
                "correlationId": correlation_id
            }
            # Remove None values from payload as Conductor might not like nulls for version/correlationId if not set
            payload = {k: v for k, v in payload.items() if v is not None}

            logger.debug(f"Starting workflow {workflow_name} with input: {workflow_input}")
            response = requests.post(url, headers=self.headers, json=payload) # Pass full payload for conductor v2+ `startWorkflow` endpoint
            
            if response.status_code == 200:
                workflow_id = response.text.strip('\"') 
                logger.info(f"Successfully started workflow {workflow_name} with ID: {workflow_id}")
                return workflow_id
            else:
                logger.error(f"Failed to start workflow {workflow_name}. Status: {response.status_code}, Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error starting workflow {workflow_name}: {e}")
            return None

    def get_workflow_status(self, workflow_id: str, include_tasks: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieves the status and details of a specific workflow instance.
        Args:
            workflow_id: The ID of the workflow instance.
            include_tasks: Whether to include detailed task information in the response.
        Returns:
            A dictionary containing the workflow status and details, or None if an error occurs.
        """
        if not workflow_id:
            logger.error("Workflow ID must be provided to get status.")
            return None
        try:
            url = f"{self.server_api_url}/workflow/{workflow_id}?includeTasks={str(include_tasks).lower()}"
            logger.debug(f"Getting status for workflow ID: {workflow_id}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting status for workflow {workflow_id}: {e}")
            return None

# Example usage (for testing this client directly):
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     # Assumes Conductor is running at http://localhost:8080/api
#     client = ConductorClient(server_api_url="http://localhost:8080/api")
    
#     if client.check_connection():
#         logger.info("Conductor server is up and running.")

        # # Example Task Definition
        # sample_task_def = {
        #     "name": "my_sample_python_task_01",
        #     "description": "A sample task implemented in Python",
        #     "retryCount": 1,
        #     "timeoutSeconds": 60,
        #     "inputKeys": ["input_value"],
        #     "outputKeys": ["output_value"],
        #     "timeoutPolicy": "TIME_OUT_WF",
        #     "retryLogic": "FIXED",
        #     "retryDelaySeconds": 5,
        #     "ownerEmail": "dev@example.com"
        # }
        # if client.register_task_definitions([sample_task_def]):
        #     logger.info("Sample task definition registered/verified.")

        # # Example Workflow Definition (using the above task)
        # sample_workflow_def = {
        #     "name": "my_sample_python_workflow_01",
        #     "description": "A sample workflow with a Python task",
        #     "version": 1,
        #     "tasks": [
        #         {
        #             "name": "my_sample_python_task_01",
        #             "taskReferenceName": "task_ref_1",
        #             "inputParameters": {"input_value": "${workflow.input.data}"},
        #             "type": "SIMPLE"
        #         }
        #     ],
        #     "inputParameters": ["data"],
        #     "outputParameters": {"result": "${task_ref_1.output.output_value}"},
        #     "ownerEmail": "dev@example.com",
        #     "schemaVersion": 2
        # }
        # if client.register_workflow_definition(sample_workflow_def):
        #     logger.info("Sample workflow definition registered/verified.")

            # # Start the workflow
            # wf_input = {"data": "Hello Conductor!"}
            # workflow_id = client.start_workflow("my_sample_python_workflow_01", wf_input)
            # if workflow_id:
            #     logger.info(f"Workflow started. Instance ID: {workflow_id}")
            #     # Check status after a short delay
            #     import time
            #     time.sleep(2)
            #     status = client.get_workflow_status(workflow_id)
            #     if status:
            #         logger.info(f"Workflow status for {workflow_id}: {status.get('status')}")
            #         # print(json.dumps(status, indent=2))
#     else:
#         logger.error("Failed to connect to Conductor server.") 