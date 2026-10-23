---
title: "Shadhra: Arabic-First Reading Companion"
slug: "shadhra"
date: "2026-10-21"
coverImage: "/images/projects/shadhra.jpg"
tags: ["Product", "Full-Stack", "OCR", "Arabic"]
excerpt: "A live web and Android app for saving quotes from printed books: photograph a page, extract the Arabic or English text, and resurface your library every day."
category: "Products"
liveDemoUrl: "https://shadhra.app"
featured: true
---

# Shadhra (شَذْرَة)

> Readers of printed books lose the lines that moved them: in phone photos, scattered notes, and margins they never reopen. Shadhra keeps them in one place.

Live on the web at [shadhra.app](https://shadhra.app) and on [Google Play](https://play.google.com/store/apps/details?id=app.shadhra).

## What it does

- **Capture:** photograph a book page, select the passage, and the text is extracted automatically (Arabic or English), then reviewed and saved with its book, author, and page.
- **Reflect and organize:** notes on every quote, books, authors, tags, collections, and links between quotes.
- **Find:** full-text and meaning-based search, plus a chat that answers from your own library.
- **Remember:** a daily quote from your archive and "on this day".
- **Share:** quote cards and short essays built from your quotes.

<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center">
  <img src="/images/projects/shadhra/ar-light-home.jpg" alt="Home screen" width="220" />
  <img src="/images/projects/shadhra/ar-light-ai-chat.jpg" alt="Chat with your library" width="220" />
  <img src="/images/projects/shadhra/ar-dark-constellation.jpg" alt="Quote map" width="220" />
</div>

## How it is built

- **Backend:** FastAPI (Python 3.12), PostgreSQL with Alembic migrations, Redis and a background worker for OCR and AI jobs.
- **Clients:** React + Vite + Tailwind web app, and an Expo / React Native Android app on one shared API.
- **AI layer:** a multi-provider router with fallbacks, per-feature cost tracking, and spending limits.
- **Quality gates:** CI runs linting, type checks, about 300 backend and web test files, and an OCR accuracy gate that fails the build if the character error rate on a fixed Arabic test set exceeds 5%.
- **Operations:** an admin console with feature flags, kill switches, and usage dashboards.

## What I learned

Building the product was the easy half. Early user research showed most new users never saved a first quote, so the latest releases focus on one thing: getting a reader from install to first saved quote in under a minute.
