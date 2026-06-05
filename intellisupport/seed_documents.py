"""
Seed documents for IntelliSupport.
Exactly 10 documents with doc_001–doc_010 and titles matching the spec exactly.
"""

SEED_DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "title": "Getting Started with Nexora",
        "source_url": "https://help.nexora.io/getting-started",
        "metadata": {"category": "general_inquiry", "version": "2024-Q4"},
        "content": (
            "Welcome to Nexora, the intelligent project management platform built for modern B2B SaaS teams. "
            "This guide will walk you through everything you need to know to get your workspace up and running, "
            "from creating your account to inviting your first team members and launching your first project.\n\n"
            "To create a Nexora account, visit app.nexora.io and click 'Start Free Trial'. Enter your work email "
            "address, choose a strong password that meets our security requirements (minimum 12 characters, "
            "at least one uppercase letter, one lowercase letter, one digit, and one special character), and "
            "verify your email address by clicking the confirmation link sent within 60 seconds. If you do not "
            "receive the confirmation email, check your spam folder or contact support@nexora.io.\n\n"
            "After email verification, you will be prompted to set up your workspace. Your workspace is the "
            "central hub for all your team's projects, documents, and conversations. Choose a workspace name "
            "that reflects your organization (e.g., 'Acme Corp Engineering'). The workspace URL slug is "
            "generated automatically from your workspace name and can be customized before creation. Once set, "
            "the workspace slug cannot be changed, so choose carefully.\n\n"
            "The Nexora onboarding wizard guides you through four steps: (1) Create your first project by "
            "selecting a template or starting from scratch. Nexora offers templates for Software Development, "
            "Marketing Campaigns, Customer Onboarding, HR Recruiting, and Event Planning. (2) Invite your team "
            "by entering one or more work email addresses. Invitees receive an email with a 72-hour invitation "
            "link and are assigned the Editor role by default; you can change this before sending. (3) Connect "
            "your tools by authorizing integrations with Slack, GitHub, or Jira if your team uses them. "
            "(4) Set up notifications to configure which events trigger Slack messages, email alerts, or "
            "in-app notifications for you and your team.\n\n"
            "Nexora's dashboard provides an at-a-glance view of all active projects, recent activity, and "
            "upcoming deadlines. The left sidebar contains navigation links to Projects, Inbox, Team, "
            "Integrations, Analytics, and Settings. The Analytics section shows query volume, response quality "
            "metrics, and feedback trends across your knowledge base. Use the search bar at the top of the "
            "dashboard to quickly find projects, documents, or responses by keyword.\n\n"
            "If you encounter any issues during onboarding, our support team is available via email at "
            "support@nexora.io (response within 24 hours on Professional and Business plans) or via the "
            "in-app chat widget available in the bottom-right corner of the dashboard. Enterprise customers "
            "also have access to a dedicated customer success manager and a private Slack channel for real-time "
            "support. The Nexora Help Center at help.nexora.io contains searchable documentation, video "
            "tutorials, and a community forum where you can ask questions and share best practices with other "
            "Nexora users."
        ),
    },
    {
        "doc_id": "doc_002",
        "title": "Managing Team Members and Permissions",
        "source_url": "https://help.nexora.io/account/team-management",
        "metadata": {"category": "account_management", "version": "2024-Q4"},
        "content": (
            "Nexora's team management system lets workspace administrators add collaborators, assign granular "
            "roles, and control access to sensitive features without sharing credentials. This article explains "
            "invitation flows, role definitions, and best practices for maintaining a secure team configuration.\n\n"
            "To invite a team member, navigate to Settings > Team > Invite Member. Enter the invitee's work "
            "email address and select one of four available roles: Viewer, Editor, Manager, or Admin. An "
            "invitation email is sent immediately; the link expires after 72 hours. If the invitee does not "
            "receive the email, ask them to check spam folders and verify their IT department has not blocked "
            "mail from nexora.io. You can resend or revoke invitations from the Pending Invitations tab.\n\n"
            "Role definitions are as follows. The Viewer role grants read-only access to the knowledge base, "
            "response history, and analytics dashboards. Viewers cannot ingest documents, submit queries "
            "through the API, or modify workspace settings. The Editor role allows document ingestion and "
            "querying but restricts access to billing and team management. The Manager role adds the ability "
            "to manage team members at or below the Manager level and export data. The Admin role has full "
            "workspace control including billing management, SSO configuration, API key generation, and the "
            "ability to promote or demote other Admins.\n\n"
            "Each workspace must maintain at least one Admin at all times. If the sole Admin needs to leave "
            "the organization, they must promote another member to Admin before their account can be removed. "
            "Attempting to remove the last Admin account surfaces an error: 'Cannot remove the last workspace "
            "administrator.'\n\n"
            "For organizations using SAML 2.0 Single Sign-On, team member provisioning can be automated via "
            "SCIM 2.0. With SCIM enabled, user accounts are created and deactivated automatically based on "
            "your identity provider's (IdP) push events. Supported IdPs include Okta, Azure Active Directory, "
            "and Google Workspace. SCIM configuration is available under Settings > Security > SCIM "
            "Provisioning and requires the Enterprise plan.\n\n"
            "Two-factor authentication (2FA) can be enforced at the workspace level by an Admin. Once enabled, "
            "all team members must enroll a TOTP authenticator app (such as Google Authenticator or Authy) "
            "before their next login. Members who have not enrolled are locked out after the grace period "
            "expires. Admins can generate a one-time bypass code for a specific member from the Team "
            "management panel in emergency situations. To add a new member to your team, go to Settings > "
            "Team > Invite Member and follow the prompts described above."
        ),
    },
    {
        "doc_id": "doc_003",
        "title": "Billing and Subscription Plans",
        "source_url": "https://help.nexora.io/billing/subscription-plans",
        "metadata": {"category": "billing", "version": "2024-Q4"},
        "content": (
            "Nexora offers four subscription tiers designed to match organizations of every size: Starter, "
            "Professional, Business, and Enterprise. Each plan is billed on a monthly or annual basis, with "
            "annual subscribers receiving a 20 percent discount compared to the equivalent monthly rate.\n\n"
            "The Starter plan costs USD 29 per month (or USD 278 annually) and supports up to 3 team members "
            "and 5,000 API calls per month. It includes access to the core knowledge-base search, basic "
            "analytics, and email support with a 48-hour response time target. The Starter plan is intended "
            "for early-stage teams evaluating the platform before committing to a larger capacity.\n\n"
            "The Professional plan is priced at USD 99 per month (or USD 950 annually) and raises the team "
            "member limit to 15 and the monthly API call quota to 50,000. Professional subscribers gain access "
            "to hybrid retrieval (BM25 plus vector search), intent classification, automated evaluation "
            "scoring, and priority email support with a 24-hour SLA.\n\n"
            "The Business plan at USD 299 per month (or USD 2,870 annually) supports up to 50 team members "
            "and 250,000 API calls per month. It adds dedicated Slack support, custom data retention policies "
            "up to 3 years, SSO via SAML 2.0, advanced audit logs, and a 99.9 percent uptime SLA backed by "
            "service credits.\n\n"
            "The Enterprise plan has custom pricing negotiated directly with the Nexora sales team and includes "
            "unlimited team members, unlimited API calls, dedicated infrastructure, a named customer success "
            "manager, custom SLA agreements, and on-premise deployment options. Organizations processing more "
            "than 1 million support interactions per month are strongly encouraged to contact the sales team "
            "for a tailored quote.\n\n"
            "Billing occurs on the anniversary date of the subscription start. Invoices are issued in PDF "
            "format and sent to the billing email address on file. Payment is accepted via major credit cards "
            "(Visa, Mastercard, American Express), ACH bank transfer for US accounts, and SEPA direct debit "
            "for EU accounts. Plan upgrades take effect immediately; the prorated difference is charged to "
            "the payment method on file. Plan downgrades take effect at the end of the current billing period. "
            "Cancellations submitted before the renewal date prevent the next charge; access continues until "
            "the end of the paid period. If you cancel your subscription, your data is retained for 60 days "
            "after the cancellation date, during which you may re-subscribe or export your data. After 60 "
            "days, all workspace data is permanently deleted. Refunds are not issued for partial months on "
            "monthly plans; annual plan refunds are evaluated on a case-by-case basis by the billing team "
            "at billing@nexora.io."
        ),
    },
    {
        "doc_id": "doc_004",
        "title": "Integrations: Slack, GitHub, and Jira",
        "source_url": "https://help.nexora.io/integrations/overview",
        "metadata": {"category": "integration", "version": "2024-Q4"},
        "content": (
            "Nexora integrates natively with Slack, GitHub, and Jira to embed AI-powered support intelligence "
            "directly into the tools your team already uses. This article covers the setup, required "
            "permissions, and key capabilities of each integration.\n\n"
            "Slack Integration: To connect Nexora to Slack, navigate to Settings > Integrations > Slack and "
            "click 'Add to Slack'. You will be redirected to Slack's OAuth consent screen. The integration "
            "requests channels:read, chat:write, commands, and users:read.email scopes. You must be a Slack "
            "Workspace Admin to authorize the installation. After installation, configure the notification "
            "channel and invite the Nexora Assistant bot to any private channels where alerts should be "
            "delivered. The /nexora slash command supports: /nexora ask [question] for hybrid retrieval "
            "queries, /nexora status for service health, and /nexora escalate [description] to create "
            "support tickets. All commands are logged to the Nexora audit trail.\n\n"
            "GitHub Integration: To connect GitHub, navigate to Settings > Integrations > GitHub and click "
            "'Connect to GitHub'. Nexora requests repo, read:org, and webhooks permissions. After "
            "authorization, select target repositories and configure label mappings: the 'technical_issue' "
            "intent maps to the 'bug' label, and 'feature_request' maps to 'enhancement'. To create a "
            "GitHub issue from a support response, click 'Escalate to GitHub' in the dashboard. The issue "
            "body is pre-populated with the query, response, chunk sources, and intent classification. "
            "Sensitive customer information is automatically redacted before transmission to GitHub. Status "
            "synchronization works via webhooks — when a GitHub issue is closed, Nexora records a status "
            "update on the corresponding response.\n\n"
            "Jira Integration: To connect Jira Cloud, navigate to Settings > Integrations > Jira and "
            "click 'Connect with Atlassian'. Nexora requests read:jira-work, write:jira-work, and "
            "read:jira-user scopes. For Jira Data Center, provide your instance URL and an API token. "
            "Configure the project mapping and issue type mapping: 'technical_issue' defaults to 'Bug', "
            "'feature_request' defaults to 'Story', and other intents default to 'Task'. Field mapping "
            "automatically populates Summary, Description, Labels, and Priority fields. Bidirectional "
            "synchronization ensures that when a Jira issue transitions to a resolved status, Nexora "
            "marks the corresponding response as resolved. Webhook delivery to Jira is retried up to "
            "5 times with exponential backoff if the Jira instance is temporarily unavailable."
        ),
    },
    {
        "doc_id": "doc_005",
        "title": "Project Templates and Workflows",
        "source_url": "https://help.nexora.io/features/templates-and-workflows",
        "metadata": {"category": "feature_request", "version": "2024-Q4"},
        "content": (
            "Nexora's project templates allow teams to standardize their workflows and eliminate repetitive "
            "setup work when starting new projects. This article explains how to use built-in templates, "
            "create custom templates, and configure automated workflow rules.\n\n"
            "Built-in Templates: Nexora ships with five production-ready project templates. The Software "
            "Development template includes columns for Backlog, In Progress, In Review, Testing, and Done, "
            "with default issue types for Bug, Feature, Task, and Epic. The Marketing Campaign template "
            "provides stages for Ideation, Content Creation, Review, Approved, and Published. The Customer "
            "Onboarding template tracks prospects through Lead, Qualified, Demo Scheduled, Contract Sent, "
            "and Closed Won stages. The HR Recruiting template manages candidates from Applied through "
            "Screened, Interviewing, Offer Extended, and Hired. The Event Planning template covers Venue "
            "Booking, Speaker Outreach, Promotion, Logistics, and Post-Event Review.\n\n"
            "Creating Custom Templates: Any existing project can be saved as a custom template. Navigate "
            "to the project settings, click 'Save as Template', and enter a template name and description. "
            "Custom templates preserve column structure, issue types, automation rules, and field "
            "configurations but do not copy individual tasks or their content. Custom templates are "
            "available to all workspace members and can be marked as private (visible only to you) or "
            "shared (visible to the entire workspace).\n\n"
            "Workflow Automation: Nexora's workflow engine supports if-then automation rules that trigger "
            "actions based on project events. Available triggers include: when an issue is moved to a "
            "specific column, when an issue's due date passes, when a label is added, and when a comment "
            "is added. Available actions include: assign the issue to a team member, send a Slack "
            "notification, create a linked GitHub issue, update a custom field value, and move the issue "
            "to another column. Automation rules are configured per project under Project Settings > "
            "Automation. Each project supports up to 50 active automation rules. Rule execution is "
            "logged and visible in the Project Activity feed.\n\n"
            "Template Marketplace: Enterprise customers have access to the Nexora Template Marketplace, "
            "where they can browse and install templates contributed by the Nexora community and verified "
            "technology partners. Marketplace templates are reviewed by the Nexora team before publication. "
            "To submit a template to the marketplace, export it from your workspace and submit the "
            "template package file via the marketplace submission form at nexora.io/marketplace/submit."
        ),
    },
    {
        "doc_id": "doc_006",
        "title": "Notifications and Alert Settings",
        "source_url": "https://help.nexora.io/features/notifications",
        "metadata": {"category": "feature_request", "version": "2024-Q4"},
        "content": (
            "Nexora's notification system keeps your team informed about important events across projects, "
            "knowledge base queries, and system health without overwhelming inboxes with irrelevant alerts. "
            "This article explains how to configure notification preferences at the workspace, project, "
            "and personal levels.\n\n"
            "Notification Channels: Nexora delivers notifications through three channels: in-app "
            "notifications (the bell icon in the dashboard header), email notifications sent to your "
            "registered address, and Slack messages if the Slack integration is configured. Each channel "
            "can be enabled or disabled independently for each notification type.\n\n"
            "Workspace-Level Notification Rules: Workspace Admins can configure global notification rules "
            "under Settings > Notifications. Available workspace-level triggers include: new low-confidence "
            "responses (confidence below a configurable threshold, default 0.7), responses with a "
            "faithfulness score below a threshold (default 0.6), feedback ratings of 1 or 2 stars, API "
            "error rate exceeding 5 percent in a rolling 5-minute window, and daily or weekly digest "
            "summaries of usage metrics. Each rule can be targeted to specific Slack channels or email "
            "distribution lists.\n\n"
            "Project-Level Notification Rules: Project managers can configure notifications scoped to a "
            "specific project under Project Settings > Notifications. Project-level triggers include: "
            "issue created, issue assigned to me, issue moved to a specific column, comment added to "
            "an issue I am watching, due date approaching (configurable advance notice: 1, 3, or 7 days), "
            "and automation rule execution failures. Project notifications are delivered to all project "
            "members whose personal preferences allow them.\n\n"
            "Personal Notification Preferences: Individual users can customize their notification "
            "preferences under Settings > Account > Notifications. You can choose to receive only "
            "mentions (@username in comments), all activity on issues you are watching, or a daily "
            "digest. The 'Do Not Disturb' schedule allows you to suppress all non-urgent notifications "
            "during specified hours (e.g., outside 9 AM to 6 PM in your local timezone). Urgent "
            "notifications — such as a critical service outage alert or a direct Admin message — "
            "bypass Do Not Disturb settings.\n\n"
            "Alert Escalation: If a critical alert (e.g., sustained high error rate or faithfulness "
            "score below 0.4 across multiple consecutive queries) is not acknowledged within 15 minutes, "
            "Nexora automatically escalates the alert to all workspace Admins via email and Slack. "
            "Escalated alerts are logged in the Alert History section of the Analytics dashboard with "
            "full context including the triggering event, timestamp, and escalation chain."
        ),
    },
    {
        "doc_id": "doc_007",
        "title": "Data Export and Backup",
        "source_url": "https://help.nexora.io/data/exports-and-backup",
        "metadata": {"category": "data_and_export", "version": "2024-Q4"},
        "content": (
            "Nexora provides structured data export capabilities to support business continuity, compliance "
            "requirements, and GDPR data subject requests. This article describes the available export "
            "formats, how to initiate exports, retention schedules, and the process for handling access "
            "and erasure requests under GDPR, CCPA, and similar privacy regulations.\n\n"
            "Initiating an Export: Workspace data can be exported from the Settings > Data Management > "
            "Export panel. Three export types are available. A full workspace export packages all "
            "documents, chunks, query history, response history, and feedback records into a ZIP archive "
            "containing one JSON Lines file per table. A selective export allows you to choose a date "
            "range and one or more data types. A single-document export downloads a specific document "
            "and all associated chunks in JSON format. All exports are generated asynchronously; you "
            "will receive an email with a signed download link valid for 24 hours once the export "
            "is ready.\n\n"
            "Export Formats: Supported export formats are JSON Lines (default, suitable for programmatic "
            "processing), CSV (for spreadsheet analysis — note that embedding vectors are excluded from "
            "CSV exports due to size constraints), and Parquet (for data warehouse ingestion, includes "
            "full vector data). Format selection is available in the Export configuration panel.\n\n"
            "Automated Backups: Nexora performs automated daily backups of all workspace data. Backups "
            "are stored in geographically redundant object storage with AES-256 encryption at rest. "
            "Business plan backups are retained for 30 days. Enterprise plan backups are retained for "
            "up to 1 year (configurable). Backups are not directly accessible by customers; they are "
            "used internally for disaster recovery. To recover accidentally deleted data, contact "
            "support@nexora.io within the backup retention window.\n\n"
            "Data Retention: The default retention period for query and response records is 90 days "
            "for Starter and Professional plans, 1 year for Business, and up to 7 years for Enterprise "
            "(configurable). Retention changes apply prospectively; records already older than the new "
            "retention period are purged within 30 days of the configuration change.\n\n"
            "GDPR and Privacy Requests: For GDPR Article 15 (access) and Article 17 (erasure) requests, "
            "submit the request form at nexora.io/privacy/dsr. Nexora will acknowledge within 72 hours "
            "and fulfill within 30 calendar days. For erasure requests, all PII is deleted from active "
            "databases and marked for purge from backup media within 90 days. A written confirmation "
            "of erasure completion will be provided. To export your project data as CSV, use the "
            "selective export option in Settings > Data Management > Export and select CSV format."
        ),
    },
    {
        "doc_id": "doc_008",
        "title": "Two-Factor Authentication Setup",
        "source_url": "https://help.nexora.io/account/two-factor-authentication",
        "metadata": {"category": "account_management", "version": "2024-Q4"},
        "content": (
            "Two-factor authentication (2FA) significantly increases account security by requiring a "
            "second verification step in addition to your password. Nexora supports TOTP-based "
            "authenticator apps and FIDO2 hardware security keys. This article covers setup, recovery "
            "options, and workspace-level enforcement.\n\n"
            "Setting Up TOTP (Authenticator App): To enroll an authenticator app, log into your Nexora "
            "account and navigate to Settings > Security > Multi-Factor Authentication. Click 'Add "
            "authenticator app' and scan the displayed QR code with a TOTP-compatible application such "
            "as Google Authenticator, Authy, Microsoft Authenticator, or 1Password. Enter the 6-digit "
            "code shown in your app to confirm enrollment. Save the 16-character backup codes displayed "
            "during setup — store them in a password manager or print and secure them physically. Each "
            "backup code is single-use and cannot be regenerated after initial display.\n\n"
            "Setting Up a Hardware Security Key: FIDO2-compliant hardware security keys (YubiKey, "
            "Google Titan Key) are supported as a second factor. Navigate to Settings > Security > "
            "Security Keys and follow the on-screen prompts. Insert or tap your security key when "
            "prompted. Up to 5 hardware keys can be registered per account, which is useful for "
            "maintaining backup keys stored in secure locations.\n\n"
            "Workspace-Level 2FA Enforcement: Workspace Admins can require all members to use 2FA "
            "by navigating to Settings > Security > Authentication and enabling 'Require Two-Factor '
            "Authentication'. Once enforced, members without 2FA enrolled receive a grace period "
            "(configurable: 24 hours to 7 days) to enroll before they are locked out. Admins can "
            "generate one-time bypass codes for individual members from the Team management panel "
            "in emergency situations.\n\n"
            "Password Reset: To reset a forgotten password, click 'Forgot Password' on the login "
            "page at app.nexora.io. Enter your account email address. A reset link is sent within "
            "60 seconds and is valid for 30 minutes. New passwords must be minimum 12 characters "
            "with at least one uppercase letter, one lowercase letter, one digit, and one special "
            "character. If your workspace enforces SSO via SAML 2.0, the password reset flow is "
            "disabled and must be performed through your identity provider.\n\n"
            "Account Recovery When MFA Device is Lost: If you are locked out due to a lost MFA "
            "device and have no backup codes, contact support@nexora.io with the subject line "
            "'MFA Account Recovery' from the email address on your account. The support team "
            "will verify your identity using billing information, last login IP, and an identity "
            "verification step before granting a one-time bypass. Admin account recovery may "
            "require a video call with the support team due to the elevated access level."
        ),
    },
    {
        "doc_id": "doc_009",
        "title": "API Access and Webhooks",
        "source_url": "https://help.nexora.io/developers/api-and-webhooks",
        "metadata": {"category": "technical_issue", "version": "2024-Q4"},
        "content": (
            "Nexora exposes a RESTful API that enables programmatic access to all platform capabilities, "
            "including document ingestion, querying, evaluation, and feedback. Webhooks allow external "
            "systems to receive real-time event notifications from Nexora without polling. This article "
            "covers API authentication, rate limits, webhook setup, and troubleshooting.\n\n"
            "API Authentication: All Nexora API requests must include an Authorization header with a "
            "Bearer token: Authorization: Bearer YOUR_API_KEY. API keys are generated in Settings > "
            "API Keys. Each key can be scoped with read-only or read-write permissions. Keys begin "
            "with the prefix 'nxr_' followed by exactly 40 alphanumeric characters. HTTP 401 is "
            "returned for missing or malformed keys. HTTP 403 is returned when the key lacks "
            "permission for the requested operation. Rotate keys every 90 days as a security best "
            "practice — generate the new key, update all services, verify authentication, then "
            "revoke the old key. Both keys remain active during a 24-hour grace period.\n\n"
            "Rate Limits: The Nexora API enforces per-minute and per-day rate limits by subscription "
            "plan. Response headers include X-RateLimit-Limit, X-RateLimit-Remaining, and "
            "X-RateLimit-Reset (Unix timestamp). HTTP 429 Too Many Requests is returned when limits "
            "are exceeded. Implement exponential backoff with jitter when handling 429 responses — "
            "retrying immediately will not succeed and may result in a temporary IP block after "
            "five consecutive 429 responses.\n\n"
            "Webhook Configuration: Webhooks deliver real-time event notifications to your endpoints. "
            "To configure a webhook, navigate to Settings > Webhooks > Add Endpoint. Enter your "
            "HTTPS endpoint URL and select the events to subscribe to. Available events include: "
            "query.completed, response.evaluated, feedback.received, ingestion.completed, and "
            "system.alert. Nexora signs each webhook payload with an HMAC-SHA256 signature using "
            "your webhook secret; verify this signature before processing the payload.\n\n"
            "Webhook Troubleshooting: If your webhook is not receiving events, check the Webhook "
            "Event Log in Settings > Webhooks for delivery status and error details. Common causes "
            "include: endpoint returning non-2xx status codes, SSL certificate errors, request "
            "timeout (Nexora waits up to 5 seconds for a response), and firewall rules blocking "
            "Nexora's IP ranges (listed at nexora.io/ip-ranges). Failed webhook deliveries are "
            "retried up to 5 times with exponential backoff (1s, 2s, 4s, 8s, 16s). After 5 "
            "failed attempts, the webhook is automatically disabled and an alert is sent to "
            "workspace Admins. Re-enable the webhook from the Webhooks settings panel after "
            "resolving the underlying issue."
        ),
    },
    {
        "doc_id": "doc_010",
        "title": "Troubleshooting Common Errors",
        "source_url": "https://help.nexora.io/technical/troubleshooting",
        "metadata": {"category": "technical_issue", "version": "2024-Q4"},
        "content": (
            "This article provides step-by-step troubleshooting guidance for the most frequently "
            "reported errors on the Nexora platform. If you cannot resolve an issue using this "
            "guide, contact support@nexora.io with your workspace ID and the error details.\n\n"
            "ERR_AUTH_401 — Unauthorized: Your API key is missing, malformed, or has been revoked. "
            "Verify the key begins with 'nxr_' followed by exactly 40 alphanumeric characters and "
            "is passed as a Bearer token in the Authorization header. Check for trailing spaces or "
            "newline characters copied from a config file. Test with: curl -H 'Authorization: "
            "Bearer YOUR_KEY' https://api.nexora.io/v1/health. If the key was recently rotated, "
            "ensure all dependent services have been updated to use the new key.\n\n"
            "ERR_AUTH_403 — Forbidden: Your API key is valid but lacks permission for the requested "
            "operation. Read-only keys cannot create, update, or delete resources. Navigate to "
            "Settings > API Keys to review and update the key's permission scope.\n\n"
            "ERR_RATE_429 — Too Many Requests: You have exceeded your plan's rate limit. Check the "
            "X-RateLimit-Remaining and X-RateLimit-Reset headers to understand when your quota "
            "resets. Implement exponential backoff in your integration code. If you consistently "
            "hit rate limits, consider upgrading to a higher plan or contacting sales about a "
            "custom rate limit increase.\n\n"
            "ERR_EMBED_503 — Embedding Service Unavailable: The OpenAI embedding endpoint is "
            "temporarily unavailable. Nexora automatically retries embedding requests up to 3 "
            "times with exponential backoff (1s, 2s, 4s). If the error persists beyond 30 seconds, "
            "check the OpenAI status page at status.openai.com and the Nexora status page at "
            "status.nexora.io. Ingestion requests will fail gracefully with this error code during "
            "an OpenAI outage; retry the ingestion once the service is restored.\n\n"
            "ERR_CHUNK_EMPTY — No Chunks Retrieved: The hybrid retrieval system returned zero "
            "results for your query. This typically means the knowledge base does not contain "
            "relevant documents for the query topic, or the similarity threshold is set too high. "
            "Solutions: ingest additional documents covering the topic, lower the "
            "similarity_threshold parameter in your query, or decrease the hybrid_alpha to give "
            "more weight to keyword (BM25) matching.\n\n"
            "ERR_DB_CONN — Database Connection Failed: Nexora cannot connect to the PostgreSQL "
            "database. Verify that your DATABASE_URL in the .env file is correct, that PostgreSQL "
            "is running and accepting connections on the specified host and port, and that the "
            "database user has the necessary privileges. If running in Docker, ensure the db "
            "container is healthy before the api container starts — the docker-compose healthcheck "
            "handles this automatically. Check pg_isready output and PostgreSQL logs for details."
        ),
    },
]
