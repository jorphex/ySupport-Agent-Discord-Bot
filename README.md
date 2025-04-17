## ySupport Agent Discord Bot

Support agent bot that answers questions by making yDaemon and RPC queries and searching Yearn documentation. 

Uses OpenAI Responses and Agents, and Pinecone. Conversation threads in ticket channels are isolated, maintaining context. Backs off and ignores further processing and response generation after user in ticket thanks the bot or says bye. Calls ySupport team when user question cannot be answered or if conversation turns reach 15. Can be triggered by a specified user to generate a single response in public channels by replying `y` for yearn context or `b` for bearn context to a user question in public channels, not maintaining context nor conversation threads. 
