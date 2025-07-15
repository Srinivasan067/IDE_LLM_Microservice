const express = require('express');
const app = express();
const port = 3000;
const bodyParser = require('body-parser');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const dotenv = require('dotenv');
const OpenAI = require("openai");

dotenv.config();
app.use(bodyParser.json());
app.use(cors());

const uri = process.env.Mongo_Url;
const client = new MongoClient(uri);

const openai = new OpenAI({
  apiKey: process.env.GITHUB_TOKEN,
  baseURL: "https://models.github.ai/inference/"
});

const forbiddenTopics = ["bmw", "ceo", "president", "weather", "capital of"];

app.post('/api/llama', async (req, res) => {
  console.log('API /api/llama was called', req.body);

  const { userId, prompt, systemMessage } = req.body;

  if (!userId || !prompt) {
    console.log('Missing userId or prompt');
    return res.status(400).json({ error: 'userId and prompt are required' });
  }

  if (forbiddenTopics.some(word => prompt.toLowerCase().includes(word))) {
    console.log('Blocked forbidden topic:', prompt);
    return res.json({
      aiResponse: "Sorry, I can only answer questions based on the provided dataset."
    });
  }

  const defaultSystemMessage = `You are DOXSY.AI, an assistant. Use ONLY the context below to answer the user's question. If the answer is not in the context, say "I don't know."`;

  try {
    const db = client.db();
    const vectors = db.collection('vectors');

    // Step 1: Embed user prompt
    const embedResponse = await openai.embeddings.create({
      input: prompt,
      model: "openai/text-embedding-3-small"
    });
    const promptEmbedding = embedResponse.data[0].embedding;

    // Step 2: Retrieve all chunks
    const allChunks = await vectors.find({}).toArray();

    // Cosine similarity
    function cosineSim(a, b) {
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    const scored = allChunks.map(doc => ({
      chunk: doc.chunk,
      score: cosineSim(promptEmbedding, doc.embedding)
    }));

    scored.sort((a, b) => b.score - a.score);

    // Filter by threshold
    const threshold = 0.75;
    const topChunks = scored
      .filter(s => s.score > threshold)
      .slice(0, 3)
      .map(s => s.chunk);

    // If no relevant data found
    if (topChunks.length === 0) {
      console.log("No relevant chunks found.");
      return res.json({
        aiResponse: "Sorry, I couldn't find the answer in the dataset."
      });
    }

    const context = topChunks.join('\n---\n');
    console.log('Top chunks selected:\n', topChunks);
    console.log('Context sent to LLaMA:\n', context);

    const messages = [
      {
        role: "system",
        content: systemMessage || defaultSystemMessage
      },
      {
        role: "system",
        content: `Here is the context:\n${context}`
      },
      {
        role: "user",
        content: `Q: ${prompt}`
      }
    ];

    // Step 3: Call LLM
    const response = await openai.chat.completions.create({
      messages,
      model: "openai/gpt-4o-mini",
      temperature: 1.0,
      max_tokens: 1000,
      top_p: 1.0
    });

    const reply = response.choices[0]?.message?.content || "I don't know.";
    return res.json({ aiResponse: reply });

  } catch (error) {
    console.error('Error in /api/llama:', error.message, error);
    return res.status(500).json({ error: 'Failed to get response from AI' });
  }
});

app.post('/api/code-analyze', async (req, res) => {
  const { language, code } = req.body;

  if (!language || !code) {
    return res.status(400).json({ error: 'language and code are required' });
  }

  try {
const prompt = `
You are a code reviewer. Analyze the following code snippet written in ${language}.

1. Is this code likely to be AI-generated or copied from an AI tool? (Answer: Yes/No/Maybe)
2. Does this code work as intended (syntactically and logically)? (Answer: Yes/No/Maybe)
3. Give a brief explanation for your answers in 1-2 sentences, using simple language suitable for beginners.

Code:
\`\`\`${language}
${code}
\`\`\`
Please respond in JSON format as:
{
  "aiGenerated": "Yes/No/Maybe",
  "works": "Yes/No/Maybe",
  "explanation": "..."
}
`;

    const response = await openai.chat.completions.create({
      messages: [
        { role: "system", content: "You are an expert code reviewer and AI detector." },
        { role: "user", content: prompt }
      ],
      model: "openai/gpt-4o-mini",
      temperature: 0.2,
      max_tokens: 500
    });

    // Try to parse the JSON from the AI's response
    let result;
    try {
      result = JSON.parse(response.choices[0].message.content);
    } catch {
      // fallback: return raw text if not valid JSON
      result = { raw: response.choices[0].message.content };
    }

    return res.json(result);
  } catch (error) {
    console.error('Error in /api/code-analyze:', error.message, error);
    return res.status(500).json({ error: 'Failed to analyze code' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

client.connect()
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));
