# Process Log

## AI tools used
Claude Opus 5.5 helped me code app.py, create eval scripts, deploy commands and secret manager. Used Nvidia Nemotron 3.5 lightning, openai GPT 5.6 sol and openai embedding model. Evaluated with RAGAS 0.4.3 and gpt 5.1.  I also used another ai chat to test the deployed app.

## Development narrative
Created a Langgraph supervisor with research summary and data agent in a single fastapi app.py. Deployed to cloud run.  Has 40 arxiv papers that were chunked into 6114 chunks of size 1000. Nvidia was used to generate cheap plan and notes. Openai used for SQL queries and difficult questions with evidence. Ran into rate limiting with gemini and no credits with openai/anthropic. Took too long with nvidia until I turned off reasoning. Learned how to properly pull corrective rag by using title and acronyms as well as rag. 

## Verification narrative
After live testing for 6 rounds I found issues. "what is the capital of australia" it answered correctly but showed a made up sql query. Answered with august because it didn't know what day it was. Used fault injection to discover bugs with citations by inputting incorrect api key. No db and no embedding model.  Was able to get the query for 15 topics to not hang and finish around 65-82 seconds (was taking 148 seconds). After adding timeouts and trace heartbeats.  By adding title and hybrid search as well as using k = 8 increased my RAGAS score from .64 to .84 for context recall. Same question gave a score of .14 and .64 when tested at different times. A wrong answer had a score of 0 when retested got a score of 1. Tested failover with a failed nvidia call which called openai. Confirmed on live app.  

## What I learned
Make sure you have timeouts on all your calls, workers and entire request. You can get a correct answer with an incorrect piece of evidence. Always pass in the current day to your model. You should always live test your application.  Your demo may show something that isn't happening in production or vice versa. More powerful models are good to use when answering hard multi agent questions that require evidence. 
