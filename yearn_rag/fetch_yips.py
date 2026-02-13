# fetch_yips.py
import requests
import json
import os
from datetime import datetime

# --- Configuration ---
SNAPSHOT_API_URL = "https://hub.snapshot.org/graphql"
# The CORRECT URL
IPFS_GATEWAY_URL = "https://ipfs.snapshot.box/ipfs/"
OUTPUT_DIR = "YIPS"

# This is the crucial list you will provide.
# Map your internal YIP number to the Snapshot proposal ID.
YIP_TO_PROPOSAL_ID_MAP = {
    73: "0xcc2a5f2bad97b551a02230975def5640c6f582d64c3c42eecfb1c6c76eea3b28",
    74: "0x3840d5b6daa3363933806c98335103c9086419b68513bd40f326f5cb0e07e9cf",
    75: "0xdb02fe93b77c6addfa9b197bb47f5b6d7779f69000210cffe54ea1fb35b91eec",
    76: "0xa07123d5f6d3eb236969c798c024098be32231d2c3a205bd197200c955baa10c",
    77: "0xe79fb2ef4f21ef1e9cc30dd1522c9751c74b631c4782bccbbeb25185d4ddae1d",
    78: "0x47c2883308fafd286697c391748c1381cf374b98cfa3af9d23d2fe79d31df6fb",
    79: "0xc7ded2863a10154b6b520921af4ada48d64d74e5b7989f98cdf073542b2e4411",
    80: "0x2a9ecea04244b83ed8f1ef6b4f62e9ee9a31d16c5ef3b52d00e3a185e78df78e",
    81: "0x6f3082db2cef3e0c254e569580d063cb14130a92d0bf1729bef342a386e419f2",
    82: "0x24ae3413fcf92bcb92cbbe8845d0684a4e8d36c7f164b4afee6c45c629f16f83",
    83: "0x872f23d57eea829e5fb0a5e0868f805efdb231d8a3c9e39820dd33432ccd629c",
    84: "0xeecd2a9ca79f9b22071d79d436a7e5ccc56593eb4c3bc8ef1b57c8389809a101",
    85: "0xa3223b388c484ea8a81b60bb88cda99f23d6d06b4b9798b4d0acafaa2207b686",
    86: "0xe2fc56f50b1c434ca2f80d07542b66b5ff035b22891c8d6ebb79afca62664d02",
    87: "0x4726de81255b4be972e0b7bb9f03fac222cfbcef9bd1e148e647be4b8b1fb47d",
    88: "0x9b3a40326411eea6c51ec389a802ed695de53961fa49f6d3525e256513d0a7f9",
}

# fetch_yips.py (continued)

# This is the GraphQL query we will send for each proposal.
PROPOSAL_QUERY = """
query Proposal($id: String!) {
  proposal(id: $id) {
    id
    ipfs
    title
    body
    state
    author
    created
    start
    end
    space {
      id
      name
    }
  }
}
"""

# fetch_yips.py (continued)

def fetch_proposal_data(proposal_id):
    """Sends a GraphQL query to the Snapshot API for a single proposal."""
    payload = {
        "query": PROPOSAL_QUERY,
        "variables": {"id": proposal_id}
    }
    try:
        response = requests.post(SNAPSHOT_API_URL, json=payload)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data for proposal {proposal_id}: {e}")
        return None

def fetch_ipfs_data(ipfs_hash):
    """Fetches the detailed proposal metadata from an IPFS gateway."""
    if not ipfs_hash:
        return None
    url = f"{IPFS_GATEWAY_URL}{ipfs_hash}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"   - ❌ Error fetching IPFS data from {url}: {e}")
        return None
    
# In fetch_yips.py

def format_proposal_as_markdown(yip_number, gql_data, ipfs_data):
    """
    Formats the combined API and IPFS data into a complete markdown file.
    """
    proposal = gql_data.get("data", {}).get("proposal")
    if not proposal:
        return None

    # --- Data Extraction and Prioritization ---
    ipfs_message = ipfs_data.get("data", {}).get("message", {}) if ipfs_data else {}

    # --- FIX IS HERE ---
    # First, get the raw title and clean it for safe inclusion in the f-string.
    raw_title = ipfs_message.get("title", proposal.get('title', ''))
    cleaned_title = raw_title.replace('"', '\\"') # Do the replacement outside the f-string
    # --- END FIX ---

    discussion_link = ipfs_message.get("discussion", "")
    author = ipfs_message.get("from", proposal.get('author', ''))
    
    created_date = datetime.utcfromtimestamp(proposal['created']).strftime('%Y-%-m-%d')
    start_date = datetime.utcfromtimestamp(proposal['start']).strftime('%Y-%m-%d')
    end_date = datetime.utcfromtimestamp(proposal['end']).strftime('%Y-%m-%d')
    
    status_map = {"closed": "Implemented", "active": "Proposed"}
    status = status_map.get(proposal['state'], proposal['state'].capitalize())
    
    space_id = proposal['space']['id']
    proposal_id = proposal['id']
    snapshot_url = f"https://snapshot.org/#/{space_id}/proposal/{proposal_id}"

    # --- Create YAML Frontmatter (Now using the cleaned title) ---
    frontmatter = f"""---
yip_number: {yip_number}
title: "{cleaned_title}"
status: "{status}"
author: "{author}"
created_date: "{created_date}"
discussion_link: "{discussion_link}"
---

"""
    
    # The body from IPFS is the most reliable source
    body = ipfs_message.get("body", proposal.get('body', ''))

    # --- Create the Appended Information Section ---
    info_section = f"""
## Information

_Source: [Snapshot]({snapshot_url})_

| Name          | Value                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| Snapshot Space | [{space_id}](https://snapshot.org/#/{space_id})                                    |
| Author        | {author}                                |
| IPFS          | {proposal['ipfs']} |
| Start date    | {start_date}                                                              |
| End date      | {end_date}                                                              |

## Results

| Result | Value             |
| ------ | ----------------- |
| Yes    | NEEDS_VOTE_TALLY  |
| No     | NEEDS_VOTE_TALLY  |

"""

    full_content = frontmatter + body + info_section
    return full_content

def main():
    """Main execution function."""
    print(f"--- Starting YIP Fetcher ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    for yip_number, proposal_id in YIP_TO_PROPOSAL_ID_MAP.items():
        print(f"Fetching YIP-{yip_number} (Proposal ID: {proposal_id})...")
        
        # Step 1: Fetch from GraphQL API
        gql_api_data = fetch_proposal_data(proposal_id)
        if not gql_api_data:
            continue

        # Step 2: Fetch from IPFS Gateway
        ipfs_hash = gql_api_data.get("data", {}).get("proposal", {}).get("ipfs")
        ipfs_api_data = fetch_ipfs_data(ipfs_hash)
        
        # Step 3: Format and Save
        markdown_content = format_proposal_as_markdown(yip_number, gql_api_data, ipfs_api_data)
        if not markdown_content:
            print(f"   ⚠️ Could not format markdown for YIP-{yip_number}.")
            continue
            
        filename = f"YIP-{yip_number}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"   ✅ Successfully saved {filename}")
        except IOError as e:
            print(f"   ❌ Error writing file {filename}: {e}")

    print("\n--- YIP Fetching Complete ---")
    print("(Optional) Manually review files to fill in vote 'Results'.")

if __name__ == "__main__":
    main()