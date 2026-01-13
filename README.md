# Connect ChatGPT to Databricks Agent Bricks via MCP

This project is adapted from [`mcp-server-hello-world`](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world) to add a tool that calls a Databricks Agent Bricks Supervisor agent.

This README describes deploying a custom MCP server with tools to call Databricks Agent Bricks endpoints
then integrating those tools with ChatGPT. The custom MCP Server will be deployed on Databricks App.

In this example, the MCP server has a tool to call a Multi Agent Supervisor endpoint that can call either a Genie space built on the NYC Taxi Trips sample dataset or a Knowledge Assistant built on the [Databricks AI Security Framework](https://www.databricks.com/resources/whitepaper/databricks-ai-security-framework-dasf)

### Requirements:

* Enable the following public previews in the Databricks workspace where the MCP server App will be deployed:
    * Databricks Apps - On-Behalf-Of User Authorization (Public Preview)
    * Mosaic AI Agent Bricks Preview (Beta)Beta
    * Knowledge Assistant (Beta) if using Knowledge Assistant bricks
* Databricks Account Admin level access

1. Create a [Databricks account login profile](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/config-profiles).

    Example from ~/.databrickscfg:
    ```
    [DBX_ACCT]
    host            = https://accounts.azuredatabricks.net
    account_id      = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    azure_tenant_id = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    ```

2. Create a Databricks OAuth App connection from the command line. This requires Account Admin role. This operation can be usually be done from the account UI but must be done in this case to specify the correct granular OAuth scopes.

    ```
    $ databricks account custom-app-integration create -p DBX_ACCT --json '{
        "name": "chatgpt-mcp-private-scoped",
        "redirect_urls": ["https://chatgpt.com/connector_platform_oauth_redirect"],
        "confidential": true,
        "scopes": ["iam.current-user:read","iam.access-control:read","serving.serving-endpoints"],
        "token_access_policy": {
            "access_token_ttl_in_minutes": 60,
            "refresh_token_ttl_in_minutes": 10080
        }
    }'
    ```

    You should get the following output:

    ```
    {  
    "client_id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  
    "client_secret":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
    "integration_id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  
    }  
    ```

    Copy and save the `client_id` and `secret_id`.

3. From your Databricks Workspace, create a new custom App.

    **Compute** -> **Apps** -> **Create a custom app**  

    **App Name:** mcp-server-chatgpt-agentbricks  

    **Description:** MCP server with access to Agent Bricks Multi-Agent Supervisor that can answer questions about NYC taxi trips via Genie or Databricks AI Security Framework via Knowledge Assistant  

    * Select **Next: Configure**  

    * Under **App resources**, select **+Add resource** and add a **Serving endpoint** resource for each endpoint your MCP server app supports a tool to call. In this example, there is only one endpoint for the multi-agent supervisor instance. From the serving endpoint drop-down, select your "mas-xxxxxxx-endpoint".  

    * Under **User authorization**, select **+Add scope** and add "Manage your model serving endpoints (serving.serving-endpoints)"   

    * Select **Create app** and wait for the app and compute to be created. There will be no code deployed yet.  

4. Update the configuration values in `server/tools.py`:

    * Set `WORKSPACE_URL` to your Databricks workspace URL (e.g., `https://adb-xxxxxxxxxxxxxxxx.xx.azuredatabricks.net`)
    * Set `MAS_ENDPOINT_NAME` to your Multi-Agent Supervisor endpoint name

5. From this project directory on your local system, sync the files from this project into your workspace for deployment in the app. The page for the newly-created app will also show this command.  

    ```{bash}
    $ databricks sync --watch . /Workspace/Users/<username>/mcp-server-chatgpt-agentbricks
    ```  

    You may need to add a `-p <profile_name>` to sync to the correct workspace.  

    When the code has been synced to the workspace, select **Deploy** in the app and select the directory where the code to deploy resides in your workspace.  

    Once deployed, the app will be up and the MCP server ready to use.

6. Verify the API scopes for the App match those in the OAuth App created earlier.  

    * In your App, select **Authorization**  

    * Under **User authorization**, the API Scopes should be:  

        | API Scopes | Details |
        |------------|---------|
        | iam.current-user:read (default) | App's default scope which allows reading the authenticated user's basic information in the IAM system. |
        | serving.serving-endpoints | Allows the app to manage model serving endpoints in Databricks. |
        | iam.access-control:read (default) | App's default scope which allows reading access control settings and permissions in the IAM system. |

7. Configure the MCP connection in ChatGPT from Settings -> Apps -> Create App. 

    * Name the custom tool (your MCP server)
    * Describe what kinds of the questions the MCP server has the ability to answer.  
    * Copy the MCP App URL from the app page and add `/mcp` to the end. It will be of the format:  
    `https://mcp-server-chatgpt-agentbricks-xxxxxxxxxxxxxxxx.xx.azure.databricksapps.com/mcp`  
    * Leave Authentication as "OAuth" and use the `client_id` and `client_secret` to complete `OAuth Client ID` and `OAuth Client Secret`.  

    Create the New ChatGPT app, add the new app as a tool in your chat and test.