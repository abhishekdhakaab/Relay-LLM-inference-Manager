-- Migration 003: Complexity prototype library for semantic classifier.
-- Embeddings are left NULL here — populated at application startup by
-- populate_prototype_embeddings() so the embedding model is only loaded once.

CREATE TABLE IF NOT EXISTS complexity_prototypes (
    id          SERIAL PRIMARY KEY,
    intent_type VARCHAR(64)  NOT NULL,
    tier        VARCHAR(16)  NOT NULL,   -- "simple" | "medium" | "complex"
    base_weight FLOAT        NOT NULL,
    prompt_text TEXT         NOT NULL,
    embedding   vector(384)             -- populated at startup, NULL until then
);

CREATE INDEX IF NOT EXISTS idx_prototypes_embedding_cos
    ON complexity_prototypes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- ── Seed data — 30 canonical prototype prompts ───────────────────────────────
-- Simple / factual_lookup (base_weight 0.10)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'factual_lookup', 'simple', 0.10, prompt_text
FROM (VALUES
    ('What is an API gateway?'),
    ('Define load balancing.'),
    ('What does REST stand for?'),
    ('What is a cache hit rate?'),
    ('Explain what a container is.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'simple' AND intent_type = 'factual_lookup');

-- Simple / factual_with_computation (base_weight 0.25)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'factual_with_computation', 'simple', 0.25, prompt_text
FROM (VALUES
    ('What is 15% of $200?'),
    ('Convert 5 miles to kilometres.'),
    ('How many seconds are in a day?')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'simple' AND intent_type = 'factual_with_computation');

-- Medium / comparative_analysis (base_weight 0.45)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'comparative_analysis', 'medium', 0.45, prompt_text
FROM (VALUES
    ('Compare SQL and NoSQL databases.'),
    ('What are the tradeoffs between REST and GraphQL?'),
    ('Explain the difference between TCP and UDP.'),
    ('Compare horizontal and vertical scaling.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'medium' AND intent_type = 'comparative_analysis');

-- Medium / summarization (base_weight 0.40)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'summarization', 'medium', 0.40, prompt_text
FROM (VALUES
    ('Summarise the key points of the CAP theorem.'),
    ('Give me an overview of how Kafka works.'),
    ('Explain the main ideas behind transformer architecture.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'medium' AND intent_type = 'summarization');

-- Medium / retrieval_dependent (base_weight 0.50)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'retrieval_dependent', 'medium', 0.50, prompt_text
FROM (VALUES
    ('What are the latest updates to the OpenAI API?'),
    ('Summarise my last three support tickets.'),
    ('What happened in the standup yesterday?')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'medium' AND intent_type = 'retrieval_dependent');

-- Complex / code_generation (base_weight 0.75)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'code_generation', 'complex', 0.75, prompt_text
FROM (VALUES
    ('Write a Python function to implement a LRU cache.'),
    ('Implement a rate limiter in Redis using the token bucket algorithm.'),
    ('Write a Dockerfile for a FastAPI application with Postgres.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'complex' AND intent_type = 'code_generation');

-- Complex / multi_step_planning (base_weight 0.85)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'multi_step_planning', 'complex', 0.85, prompt_text
FROM (VALUES
    ('Design a system that handles 1 million requests per second.'),
    ('How would you architect a multi-tenant SaaS application?'),
    ('Design the data model for a ride-sharing application.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'complex' AND intent_type = 'multi_step_planning');

-- Complex / mathematical_reasoning (base_weight 0.80)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'mathematical_reasoning', 'complex', 0.80, prompt_text
FROM (VALUES
    ('What is 23% of $4,847 and how does that compare to last quarter?'),
    ('Given a 15% monthly churn rate, model user retention over 12 months.'),
    ('Calculate the compound annual growth rate if revenue grew from $2M to $8M over 4 years.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'complex' AND intent_type = 'mathematical_reasoning');

-- Complex / multi_hop_reasoning (base_weight 0.90)
INSERT INTO complexity_prototypes (intent_type, tier, base_weight, prompt_text)
SELECT 'multi_hop_reasoning', 'complex', 0.90, prompt_text
FROM (VALUES
    ('Explain how RAFT consensus works, then describe how it handles network partitions, and write a pseudocode implementation of the leader election.'),
    ('Compare the memory usage of Python lists vs NumPy arrays, calculate the ratio for 1M elements, and write a benchmark script.'),
    ('Summarise ticket #2847, determine if the issue was resolved, and if not draft a follow-up message to the engineering team.')
) AS t(prompt_text)
WHERE NOT EXISTS (SELECT 1 FROM complexity_prototypes WHERE tier = 'complex' AND intent_type = 'multi_hop_reasoning');
