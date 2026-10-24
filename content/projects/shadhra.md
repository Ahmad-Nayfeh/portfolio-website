---
title: "Shadhra: Arabic-First Reading Companion"
slug: "shadhra"
date: "2026-10-21"
coverImage: "/images/projects/shadhra.jpg"
tags: ["Full-Stack", "OCR", "LLMs", "Arabic"]
excerpt: "A web and Android app for saving quotes from printed books: photograph a page, extract the Arabic or English text, and resurface your library every day."
category: "Products"
liveDemoUrl: "https://shadhra.app"
status: "Live"
featured: true
gallery:
  - "/images/projects/shadhra/library.jpg"
  - "/images/projects/shadhra/home.jpg"
  - "/images/projects/shadhra/capture.jpg"
highlights:
  - "Live on the web (shadhra.app) and on Google Play"
  - "Photo-to-text capture for Arabic and English book pages"
  - "FastAPI backend shared by a React web app and an Expo Android app"
  - "OCR accuracy gate and about 300 test files in CI"
role: "Designed, built, and released it on my own"
result: "Live on the web (shadhra.app) and on Google Play"
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


## How it is built

- **Backend:** FastAPI (Python 3.12), PostgreSQL with Alembic migrations, Redis and a background worker for OCR and AI jobs.
- **Clients:** React + Vite + Tailwind web app, and an Expo / React Native Android app on one shared API.
- **AI layer:** a multi-provider router with fallbacks, per-feature cost tracking, and spending limits.
- **Quality gates:** CI runs linting, type checks, about 300 backend and web test files, and an OCR accuracy gate that fails the build if the character error rate on a fixed Arabic test set exceeds 5%.
- **Operations:** an admin console with feature flags, kill switches, and usage dashboards.

## What I learned

Building the product was the easy half. Early user research showed most new users never saved a first quote, so the latest releases focus on one thing: getting a reader from install to first saved quote in under a minute.
