# Yearn Finance - Internal AI Context Document
_This document provides internal context for AI agents to improve their analysis and suggestions. It is not for public consumption._

## Teams and Responsibilities

### Docs
- **Ownership:** All public-facing documentation, including the docs site, FAQ page, and user guides.
- **Primary Tasks:** Writing new content, updating existing articles for clarity, ensuring technical accuracy.
- **Keywords:** documentation, docs, FAQ, guide, articles, content.

### Products
- **Ownership:** All on-chain smart contracts, including Vaults, Strategies, yLockers, yCRV, veYFI
- **Primary Tasks:** Developing new vaults and strategies, fixing smart contract bugs, managing product deprecations, monitoring on-chain performance.
- **Keywords:** smart contract, vault, strategy, on-chain, bug, performance, APY calculation logic, deprecation, veYFI, yCRV.

### Web
- **Ownership:** The frontend user interface (website) and the APIs that power it.
- **Primary Tasks:** Building new UI components, fixing frontend bugs, improving UX, ensuring data displayed on the site (APY, TVL) is accurate.
- **Keywords:** UI, UX, frontend, website, interface, tooltip, button, API, data display, APY display.

### Support
- **Ownership:** Real-time user interaction and community management.
- **Primary Tasks:** Answering user questions in Discord, managing the ySupport bot, writing and posting announcements, identifying user sentiment, resolving technical user issues.
- **Keywords:** support, tickets, ySupport, bot training, announcements, community.

## Key Products and Known Issues

### veYFI Staking
- **Description:** The primary staking mechanism for the YFI token. Users lock YFI to receive veYFI. Reward token is dYFI, Discount YFI, which can be redeemed for YFI at a discount by exchanging ETH.
- **Known User Confusion:** Users are often confused about the reward distribution mechanism, specifically that rewards are distributed as dYFI tokens which must be claimed. They also ask why their boost doesn't update instantly. Sometimes, the dYFI Reward Pool contract runs out of dYFI tokens before the team responsible for it tops it up, leaving users confused why claiming dYFI does not work.
### dYFI Redemption Contract
- **Description:** The smart contract where users redeem their dYFI rewards for YFI.
- **Known Issue:** This contract is **manually refilled** by the team on an ad-hoc basis. It can sometimes be empty, which causes users to believe the system is broken. This is a common source of "bug reports" that are actually just operational delays.
### Vault APY Display
- **Description:** The website UI displays APY for each vault. This figure is composed of multiple sources (base yield, rewards, etc.).
- **Existing Feature:** There is a tooltip on the APY figure that provides a detailed breakdown.
- **Known User Behavior:** A significant number of users do not see or click on this tooltip and ask for the breakdown in Discord.
### Vault Deprecation
- **Description:** Individual vault pages on the website UI may show banners describing the state of a deprecated vault. This can sometimes include a call to action to migrate to a newer vault if available, but withdrawing is also always an option for the user. Since contracts are immutable, most of these old vaults continue to earn yield no matter how minimal.
- **Known User Behavior:** Many users ask about their old or deprecated vault deposits before trying to find their vault deposits on the website, leading to many inquiries about finding old vaults.
### Leveraged Farming
- **Description:** Advanced users can take their Yearn vault tokens (yvTokens) and use them as collateral on external lending platforms like Morpho to borrow and create leveraged positions.
- **Known User Confusion:** Users are frequently unsure if they still earn Yearn yield and external rewards when their yvTokens are deposited in Morpho. The answer is yes, but this is not well-documented.
## ySupport Bot Capabilities
- **Function:** The ySupport bot is a powerful support bot for Yearn, with access to Yearn APIs for vault data, web3 APIs for blockchain data, and RAG (Retrieval-Augmented Generation) for Yearn documentation.
- **Knowledge Source:** It has access to the entire public documentation site (FAQ, user guides, developer documentation), Yearn APIs, and web3 APIs.
- **Current Limitation:** The bot is not yet trained to answer questions about our internal context (like new tools and products) or real-time on-chain data for a few newly deployed contracts. It can only report what is written in the docs, and what is available in the Yearn APIs and web3 APIs.

