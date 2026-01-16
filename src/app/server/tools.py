"""
Tools module for the MCP server.

This module defines all the tools (functions) that the MCP server exposes to clients.
Tools are the core functionality of an MCP server - they are callable functions that
AI assistants and other clients can invoke to perform specific actions.

Each tool should:
- Have a clear, descriptive name
- Include comprehensive docstrings (used by AI to understand when to call the tool)
- Return structured data (typically dict or list)
- Handle errors gracefully
"""

import os

from openai import OpenAI

from server import utils

# Agent tool configuration
# DATABRICKS_HOST is automatically set by Databricks Apps runtime
# AGENT_ENDPOINT_NAME and AGENT_DESCRIPTION are set in app.yaml
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
AGENT_ENDPOINT_NAME = os.environ.get("AGENT_ENDPOINT_NAME", "")
AGENT_DESCRIPTION = os.environ.get("AGENT_DESCRIPTION", "Ask questions to the AI agent")


def load_tools(mcp_server):
    """
    Register all MCP tools with the server.

    This function is called during server initialization to register all available
    tools with the MCP server instance. Tools are registered using the @mcp_server.tool
    decorator, which makes them available to clients via the MCP protocol.

    Args:
        mcp_server: The FastMCP server instance to register tools with. This is the
                   main server object that handles tool registration and routing.

    Example:
        To add a new tool, define it within this function using the decorator:

        @mcp_server.tool
        def my_new_tool(param: str) -> dict:
            '''Description of what the tool does.'''
            return {"result": f"Processed {param}"}
    """

    @mcp_server.tool
    def health() -> dict:
        """
        Check the health of the MCP server and Databricks connection.

        This is a simple diagnostic tool that confirms the server is running properly.
        It's useful for:
        - Monitoring and health checks
        - Testing the MCP connection
        - Verifying the server is responsive

        Returns:
            dict: A dictionary containing:
                - status (str): The health status ("healthy" if operational)
                - message (str): A human-readable status message

        Example response:
            {
                "status": "healthy",
                "message": "Custom MCP Server is healthy and connected to Databricks Apps."
            }
        """
        return {
            "status": "healthy",
            "message": "Custom MCP Server is healthy and connected to Databricks Apps.",
        }

    @mcp_server.tool
    def get_current_user() -> dict:
        """
        Get information about the current authenticated user.

        This tool retrieves details about the user who is currently authenticated
        with the MCP server. When deployed as a Databricks App, this returns
        information about the end user making the request. When running locally,
        it returns information about the developer's Databricks identity.

        Useful for:
        - Personalizing responses based on the user
        - Authorization checks
        - Audit logging
        - User-specific operations

        Returns:
            dict: A dictionary containing:
                - display_name (str): The user's display name
                - user_name (str): The user's username/email
                - active (bool): Whether the user account is active

        Example response:
            {
                "display_name": "John Doe",
                "user_name": "john.doe@example.com",
                "active": true
            }

        Raises:
            Returns error dict if authentication fails or user info cannot be retrieved.
        """
        try:
            w = utils.get_user_authenticated_workspace_client()
            user = w.current_user.me()
            return {
                "display_name": user.display_name,
                "user_name": user.user_name,
                "active": user.active,
            }
        except Exception as e:
            return {"error": str(e), "message": "Failed to retrieve user information"}

    # Define ask_agent with dynamic docstring from AGENT_DESCRIPTION
    def ask_agent(prompt: str) -> dict:
        """Placeholder docstring - replaced dynamically."""
        try:
            # Get the user's OBO token
            token = utils.get_user_token()

            if token is None:
                return {
                    "error": "No OBO token available",
                    "message": "This tool requires OBO authentication. Running locally without token.",
                }

            # Validate configuration
            if not DATABRICKS_HOST:
                return {
                    "error": "DATABRICKS_HOST not configured",
                    "message": "The DATABRICKS_HOST environment variable is not set. This should be automatic in Databricks Apps.",
                }
            if not AGENT_ENDPOINT_NAME:
                return {
                    "error": "AGENT_ENDPOINT_NAME not configured",
                    "message": "The AGENT_ENDPOINT_NAME environment variable is not set.",
                }

            # Create OpenAI client pointing to Databricks serving endpoints
            # Ensure DATABRICKS_HOST has https:// prefix
            host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("https://") else f"https://{DATABRICKS_HOST}"
            base_url = f"{host}/serving-endpoints"
            client = OpenAI(
                api_key=token,
                base_url=base_url,
            )

            # Call the agent using responses.create() API
            response = client.responses.create(
                model=AGENT_ENDPOINT_NAME,
                input=[{"role": "user", "content": prompt}],
            )

            # Extract text from response.output[].content[].text
            if hasattr(response, "output") and response.output:
                texts = []
                for output in response.output:
                    if hasattr(output, "content"):
                        for item in output.content:
                            if hasattr(item, "text") and item.text:
                                texts.append(item.text)
                if texts:
                    return {"response": " ".join(texts).strip()}

            # Fallback: return raw response for debugging
            return {
                "response": str(response),
                "note": "Could not extract text from response",
            }

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "401" in error_msg:
                return {
                    "error": error_msg,
                    "message": "Authentication failed. Check that the App has serving scopes and user has Can Query permission.",
                }
            if "404" in error_msg:
                return {
                    "error": error_msg,
                    "message": f"Endpoint '{AGENT_ENDPOINT_NAME}' not found or not accessible.",
                }
            # Normalize host for debug output
            debug_host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("https://") else f"https://{DATABRICKS_HOST}"
            return {
                "error": error_msg,
                "message": "Failed to query the agent",
                "debug": {
                    "base_url": f"{debug_host}/serving-endpoints",
                    "endpoint": AGENT_ENDPOINT_NAME,
                },
            }

    # Set the docstring dynamically from AGENT_DESCRIPTION environment variable
    ask_agent.__doc__ = f"""{AGENT_DESCRIPTION}

    Args:
        prompt: The question or message to send to the agent.

    Returns:
        dict: The agent's response or an error message.
    """

    # Register the tool with the MCP server
    mcp_server.tool()(ask_agent)

