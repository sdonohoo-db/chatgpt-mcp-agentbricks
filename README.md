# Connect ChatGPT to Databricks Agent Bricks via MCP

This project is adapted from [`mcp-server-hello-world`](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world) to add a tool that calls a Databricks Agent Bricks agent.

This README describes deploying a custom MCP server with tools to call Databricks Agent Bricks endpoints, then integrating those tools with ChatGPT. The MCP server is deployed as a Databricks App using Databricks Asset Bundles.

## Requirements

* Enable the following public previews in the Databricks workspace where the MCP server App will be deployed:
    * Databricks Apps - On-Behalf-Of User Authorization (Public Preview)
    * Mosaic AI Agent Bricks Preview (Beta)
* Databricks Account Admin level access
* [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed and configured

## Setup Instructions

### 1. Create a Databricks Account Login Profile

Create a [Databricks account login profile](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/config-profiles).

Example from `~/.databrickscfg`:
```
[DBX_ACCT]
host            = https://accounts.azuredatabricks.net
account_id      = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
azure_tenant_id = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### 2. Create a Databricks OAuth App

Create a Databricks OAuth App connection from the command line. This requires Account Admin role. This operation can usually be done from the account UI but must be done via CLI in this case to specify the correct granular OAuth scopes.

```bash
databricks account custom-app-integration create -p DBX_ACCT --json '{
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

```json
{
  "client_id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "client_secret":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "integration_id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

Copy and save the `client_id` and `client_secret`.

### 3. Configure the Bundle

Update the configuration with your agent endpoint:

**In `databricks.yml`:**
* Set the `agent_endpoint_name` variable to your Databricks serving endpoint
* Update the `workspace.host` for each target to your Databricks workspace URL

```yaml
variables:
  agent_endpoint_name:
    default: my-agent-endpoint

targets:
  dev:
    workspace:
      host: https://adb-xxxxxxxxxxxx.xx.azuredatabricks.net/
```

**In `src/app/app.yaml`:**
* Set `AGENT_ENDPOINT_NAME` to match the endpoint name in databricks.yml
* Set `AGENT_DESCRIPTION` to describe what the agent does (shown to AI clients like ChatGPT)

```yaml
env:
  - name: AGENT_ENDPOINT_NAME
    value: my-agent-endpoint
  - name: AGENT_DESCRIPTION
    value: Ask questions about your data
```

> **Note:** DAB variable substitution does not work in app.yaml files. The endpoint name must be configured in both databricks.yml (for resource permissions) and app.yaml (for runtime configuration).

Ensure you have authenticated with the Databricks CLI (`databricks auth login`) before deploying.

### 4. Deploy the App

Deploy the MCP server app using Databricks Asset Bundles:

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to the dev target (default)
databricks bundle deploy

# Or deploy to a specific target
databricks bundle deploy -t prod
```

The bundle will automatically:
- Create the Databricks App
- Configure the serving endpoint resource
- Deploy the application code

Once deployed, the app will be up and the MCP server ready to use.

### 5. Verify API Scopes

Verify the API scopes for the App match those in the OAuth App created earlier.

* In your Databricks workspace, navigate to **Compute** → **Apps** → your app
* Select **Authorization**
* Under **User authorization**, the API Scopes should be:

| API Scopes | Details |
|------------|---------|
| iam.current-user:read (default) | App's default scope which allows reading the authenticated user's basic information in the IAM system. |
| serving.serving-endpoints | Allows the app to manage model serving endpoints in Databricks. |
| iam.access-control:read (default) | App's default scope which allows reading access control settings and permissions in the IAM system. |

### 6. Configure ChatGPT

Configure the MCP connection in ChatGPT from **Settings** → **Apps** → **Create App**.

* Name the custom tool (your MCP server)
* Describe what kinds of questions the MCP server has the ability to answer
* Copy the MCP App URL from the app page and add `/mcp` to the end. It will be of the format:
  `https://mcp-chatgpt-agentbricks-xxxxxxxxxxxxxxxx.xx.azure.databricksapps.com/mcp`
* Leave Authentication as "OAuth" and use the `client_id` and `client_secret` to complete `OAuth Client ID` and `OAuth Client Secret`

Create the new ChatGPT app, add the new app as a tool in your chat, and test.

## Project Structure

```
databricks.yml          # Databricks Asset Bundle configuration
src/app/                # Application source code
├── app.yaml            # Databricks App configuration
├── pyproject.toml      # Python dependencies and build config
├── server/             # MCP server code
│   ├── app.py          # FastAPI + FastMCP setup
│   ├── main.py         # Entry point
│   ├── tools.py        # MCP tool definitions
│   └── utils.py        # Databricks auth helpers
├── scripts/dev/        # Developer utilities
└── tests/              # Integration tests
```

## Local Development

See `CLAUDE.md` for detailed development instructions, including:
- Adding new MCP tools
- Running integration tests
- Local server development
- Remote testing with OAuth
