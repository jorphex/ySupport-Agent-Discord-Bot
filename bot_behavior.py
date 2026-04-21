import config

LISTING_DENIAL_MESSAGE = (
    "Thank you for your interest! "
    "Yearn Finance ($YFI) is permissionlessly listable on exchanges. Yearn does not pay listing fees, nor does it provide liquidity for exchange listings. "
    "No proposal is necessary for listing.\n\n"
    "No follow up inquiries or responses necessary."
)

STANDARD_REDIRECT_MESSAGE = (
     f"Thank you for your interest! "
     f"For partnership, marketing, or other business development proposals, go to <#{config.PR_MARKETING_CHANNEL_ID}>, share your proposal in **5 sentences** describing how it benefits both parties, and tag **corn**.\n\n"
     f"No follow up inquiries or responses necessary."
)

SECURITY_VENDOR_BOUNDARY_MESSAGE = (
    "Thanks for flagging this. This support bot does not handle vendor scans, takedown offers, or security-service trials.\n\n"
    "If you are disclosing a concrete Yearn security issue or phishing infrastructure, use Yearn's official security process and contacts in https://github.com/yearn/yearn-security/blob/master/SECURITY.md .\n\n"
    "We cannot engage on vendor offerings or trial coordination here."
)

JOB_INQUIRY_REDIRECT_MESSAGE = (
    "Thank you for your interest in contributing to or working with Yearn!\n\n"
    "Yearn operates with project-based grants. You can find full details about the process in the [Yearn Docs](https://docs.yearn.finance/contributing/operations/budget).\n"
    "You may also work on open issues, report bugs, suggest improvements, write documentation, and more by visiting our [GitHub repository](https://github.com/yearn), where anyone is welcome to contribute.\n\n"
    "No follow up inquiries or responses necessary."
)

OUT_OF_SCOPE_SUPPORT_MESSAGE = (
    "This bot only handles Yearn support. Keep the request limited to Yearn vaults, deposits or withdrawals, "
    "transactions, rewards, docs, bugs, or other Yearn product behavior.\n\n"
    "No follow up inquiries or responses necessary."
)
