This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Purpose

This `webui` provides a simple UI to test the local IAM Policy Generator pipeline. It exposes an API route `/api/generate` that spawns your Python generator (`src/policy_generator.py`) and returns the generated `iam_policy` and `test_config` as JSON.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

## Prerequisites

- Python 3 available on PATH as `python3`
- Python project dependencies installed and data/indexes prepared (per repo READMEs)
- Environment variables set for the Python pipeline (in your shell or a `.env` at repo root):
  - `GEMINI_API_KEY`, `PINECONE_API_KEY` (and optionally `PINECONE_CLOUD`, `PINECONE_REGION`)

## Usage

1. Start the dev server as above
2. Submit a query on the home page; optionally select services, set score threshold, max actions, and toggle query expansion
3. The API will execute the Python generator and return results for display

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
