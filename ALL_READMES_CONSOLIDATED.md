# Consolidated README Files - Badal Sharma's Repositories

**Generated:** June 2026  
**Total Repositories:** 13 (11 Public, 1 Private)  
**Total README Files:** 12

---

## Table of Contents

1. [AI Hallucination Detector](#ai-hallucination-detector)
2. [Assignment RAG](#assignment-rag)
3. [Badal Sharma Profile](#badal-sharma-profile)
4. [Forex Trading Bot](#forex-trading-bot)
5. [Perfect Plants SKU](#perfect-plants-sku)
6. [Project Ride](#project-ride)
7. [RAG Based Searching System](#rag-based-searching-system)
8. [Shoppers Store Upsell](#shoppers-store-upsell)
9. [SpendWise](#spendwise)
10. [Thank You Upsell AI](#thank-you-upsell-ai)
11. [Hatchways - Filtering API](#hatchways---filtering-api)

---

## AI Hallucination Detector

**Repository:** https://github.com/badalsharma9929/ai-hallucination-detector

### Overview
A production-ready tool to detect potential hallucinations in AI-generated text using multiple detection methods.

### Features
- **Perplexity Analysis** - Measures how unusual/unnatural the text sounds using GPT-2
- **AI Text Detection** - Detects AI-generated text using RoBERTa (OpenAI detector)
- **Self-Consistency** - Verifies internal consistency by generating multiple responses (Groq API)
- **Fact Verification** - Cross-references facts with Wikipedia API

### Tech Stack
- **Perplexity**: GPT-2 (HuggingFace, free, CPU)
- **AI Text Detection**: roberta-base-openai-detector (HuggingFace, free, CPU)
- **LLM**: Groq API (Llama-3.1-8B-Instant, free tier)
- **Fact Check**: Wikipedia API (free)
- **Frontend**: Streamlit
- **Backend**: FastAPI ready

### Installation
```bash
git clone https://github.com/badalsharma9929/ai-hallucination-detector.git
cd ai-hallucination-detector
pip install -r requirements.txt
```

### Usage
```bash
streamlit run app.py
```

### Live Demo
https://ai-hallucination-detector-9929.streamlit.app/

---

## Assignment RAG

**Repository:** https://github.com/badalsharma9929/assignment-rag

### Overview
A local Retrieval-Augmented Generation (RAG) system that simulates Google Cloud Vertex AI behavior using open-source components. Deployed on Streamlit Cloud for easy access.

### Live Demo
https://assignment-rag-2424.streamlit.app/

### Architecture Overview
```
User Query → Embedding Model → Vector Search (FAISS) → Ranked Results → Display
                ↕
         Document Corpus
         (75 synthetic docs)
```

### Key Components
1. **Synthetic Data Generation** - Created 75 realistic documents about AI/ML topics
2. **Embedding Model** - Uses all-MiniLM-L6-v2 from Hugging Face Transformers
3. **FAISS Vector Store** - Fast similarity search with IndexFlatIP
4. **Retrieval Pipeline** - Encodes queries and searches FAISS index
5. **Benchmark Evaluation** - Tests against 25 ground-truth query-document pairs
6. **Web UI** - Built with Streamlit for interactive dashboard

### Tech Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.9+ | Core development |
| Embeddings | all-MiniLM-L6-v2 (Transformers) | Converts text → vectors |
| Vector Store | FAISS (faiss-cpu) | Fast similarity search |
| GCP Mock | unittest.mock pattern | Simulates Vertex AI |
| Web UI | Streamlit | Interactive dashboard |
| Testing | pytest | Unit tests |

### Metrics
| Metric | Local (all-MiniLM-L6-v2) | GCP Mock | Meaning |
|--------|--------------------------|----------|---------|
| Precision@5 | 0.28 | 0.02 | 28% of top-5 results are relevant |
| Recall@5 | 1.00 | 0.08 | All relevant docs found in top-5 |
| MRR | 0.85 | 0.03 | First relevant result is near the top |
| NDCG@5 | 0.90 | 0.04 | High quality ranking |

### Setup
```bash
git clone https://github.com/badalsharma9929/assignment-rag.git
cd assignment-rag
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Badal Sharma Profile

**Repository:** https://github.com/badalsharma9929/badalsharma9929

### Professional Profile
Full Stack Developer and AI/ML Specialist with expertise in building intelligent, scalable applications. Currently focusing on **Full-Stack Development (FDE)** and **AI/ML Systems**.

### Core Expertise

#### Full-Stack Development (FDE)
- **Frontend**: TypeScript, React, Vue.js, responsive UI/UX design
- **Backend**: Node.js, Python, RESTful APIs, microservices
- **Databases**: SQL, NoSQL, vector databases
- **DevOps**: Cloud deployment, CI/CD pipelines, scalability
- **E-Commerce**: Shopify platform, post-purchase automation, revenue optimization

#### Artificial Intelligence & Machine Learning
- **RAG Systems**: Retrieval-Augmented Generation, vector search optimization
- **LLM Integration**: Prompt engineering, fine-tuning, output validation
- **NLP**: Hallucination detection, semantic analysis, embeddings
- **Vector Databases**: FAISS, embeddings optimization, similarity search
- **Production AI**: Model deployment, inference optimization, monitoring

### Flagship Projects

#### AI & Machine Learning
- **AI Hallucination Detector** - Intelligent system detecting and mitigating hallucinations in LLM outputs
- **RAG Vector Search System** - High-performance retrieval-augmented generation with FAISS

#### Full-Stack Development
- **Perfect Plants SKU** - Enterprise-grade Shopify storefront with TypeScript
- **Thank You Upsell AI** - AI-powered post-purchase optimization platform
- **Shoppers Store Upsell System** - Retail e-commerce optimization platform
- **Project Ride (Fery)** - Full-stack ride-sharing application (2025 Batch)

### Technology Arsenal
```
Primary Languages: JavaScript, TypeScript, Python
Secondary: Dart, Node.js, SQL

AI/ML Stack: LLM APIs, Transformers, FAISS, sentence-transformers, NumPy, Pandas

Full-Stack: React, Vue.js, Express.js, FastAPI, Django, PostgreSQL, MongoDB, Vector DBs
```

---

## Forex Trading Bot

**Repository:** https://github.com/badalsharma9929/forex-trading-bot

### Overview
A complete forex trading bot with AI/ML capabilities, sentiment analysis, and paper trading simulation. Built with 100% free APIs - no API keys required!

### Live Dashboard
https://forex-trading-bot-jhvewmfrnsjwxawihzksx2.streamlit.app/

### Features
- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA, ADX, Stochastic, CCI, ATR
- **Sentiment Analysis**: VADER for news-based sentiment analysis
- **ML Models**: Random Forest, XGBoost framework ready
- **Risk Management**: Position sizing, stop-loss, kill switch, drawdown limits
- **Paper Trading**: Virtual ₹100,000 starting balance
- **Dashboard**: Streamlit web interface with real-time charts

### Trading Pairs (India Compliant)
- USD/INR - US Dollar / Indian Rupee
- EUR/INR - Euro / Indian Rupee
- GBP/INR - British Pound / Indian Rupee
- JPY/INR - Japanese Yen / Indian Rupee

### Tech Stack
| Component | Technology | Cost |
|-----------|-------------|------|
| Language | Python 3.10+ | Free |
| Price Data | Frankfurter API | Free |
| News Data | Sample forex news | Free |
| Sentiment | VADER (NLTK) | Free |
| Dashboard | Streamlit | Free |
| Database | SQLite | Free |
| Deployment | Streamlit Cloud | Free |

### Quick Start
```bash
git clone https://github.com/badalsharma9929/forex-trading-bot.git
cd forex-trading-bot
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Risk Management
- **Max 2% risk per trade** - Limits losses on single trades
- **Max 10% position size** - Prevents over-concentration
- **20% max drawdown** - Kill switch when losses exceed 20%
- **5% daily loss limit** - Auto-stops daily trading
- **5 consecutive losses** - Auto-stop after 5 losses in a row

---

## Perfect Plants SKU

**Repository:** https://github.com/badalsharma9929/Perfect-Plants-SKU

### Overview
Perfect Plants SKU is a production-style Shopify embedded SaaS demo for D2C brands that want better post-purchase and bundle revenue.

### Live Website
https://perfect-plants-sku.vercel.app/shopping

### What It Does
- Shows a premium home decor storefront inspired by Flipkart/Amazon product browsing
- Organizes products into three fast-shopping sections
- Includes 15 curated SKUs, five products per section
- Full product pages with four images, description, specs, pricing, trust badges
- Cart drawer and cart page
- Bundle offer popup at checkout with heavy discount
- Updates dashboard, analytics, and revenue charts in real-time
- Attributes analytics to exact purchased SKU and bundle outcome
- Syncs storefront analytics across browser tabs with localStorage and BroadcastChannel
- Includes Shopify Thank You page Checkout UI Extension
- Backend APIs for campaign matching, discounts, analytics, and webhooks

### Tech Stack
- **Frontend**: Next.js App Router, React, TypeScript, TailwindCSS, Shopify Polaris, Shopify App Bridge
- **Storefront**: React client components, local cart state, realtime BroadcastChannel analytics
- **Backend**: Node.js, NestJS, Prisma ORM
- **Database**: PostgreSQL-ready Prisma schema
- **Shopify**: Checkout UI Extensions, Admin APIs, app webhooks
- **Deployment**: Vercel for frontend, Railway-ready backend config
- **CI**: GitHub Actions typecheck and frontend build workflow

### Project Structure
```
├── .github/workflows/ci.yml
├── api/
├── backend/
│   └── src/
│       ├── analytics/
│       ├── auth/
│       ├── campaigns/
│       ├── offers/
│       └── webhooks/
├── extensions/
│   └── post-purchase-offer/
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/
│       ├── hooks/
│       └── lib/
└── prisma/
```

### Key Files
- `frontend/src/components/ShoppingRealtimeClient.tsx` - Customer shopping flow
- `frontend/src/lib/storefront-products.ts` - Storefront sections and products
- `backend/src/offers/offer-engine.service.ts` - Offer matching engine
- `extensions/post-purchase-offer/src/Checkout.tsx` - Thank You page block

### Local Setup
```bash
npm install
cp .env.example .env
npm run prisma:generate
npm run prisma:migrate
npm run db:seed
npm run dev
```

---

## Project Ride

**Repository:** https://github.com/badalsharma9929/project-ride

### Overview
A simple REST API for managing users and rides. Fery - Assignment Submission Link 2025 Batch.

### Tech Stack
- **Language**: JavaScript
- **Framework**: Node.js
- **Description**: Full-stack ride-sharing application with complete end-to-end architecture

### Setup
```bash
npm install
npm run dev
```

Server runs on port 3000.

### API Documentation

#### Authentication
**Register User**
- **POST** `/api/users/register`
- **Body**:
  ```json
  {
    "username": "john_doe",
    "password": "secure123"
  }
  ```

#### Rides
**List Rides**
- **GET** `/api/rides`

**Get Ride Details**
- **GET** `/api/rides/:id`

### Error Handling
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error

---

## RAG Based Searching System

**Repository:** https://github.com/badalsharma9929/RAG-based-searching-system

### Overview
A local Retrieval-Augmented Generation (RAG) system that simulates Google Cloud Vertex AI behavior using open-source components. This project demonstrates semantic search with vector embeddings.

### Features
- **Local Embeddings**: Uses all-MiniLM-L6-v2 - fast, lightweight sentence embeddings
- **Vector Search**: FAISS-powered fast similarity search
- **GCP Simulation**: Mocks Vertex AI's `textembedding-gecko` using unittest.mock
- **Benchmark Metrics**: Precision@K, Recall@K, MRR, NDCG@K
- **Fully Local**: No API keys, no external services needed

### Quick Start
```bash
git clone https://github.com/badalsharma9929/RAG-based-searching-system.git
cd RAG-based-searching-system
pip install -r requirements.txt
python main.py
```

### Output Files
- `output/results.json` - Full metrics in JSON
- `output/retrieval_benchmark.md` - Markdown comparison table

### Project Structure
```
rag-assessment/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── benchmark.py
│   ├── data_loader.py
│   └── generate_data.py
├── tests/
├── data/
│   ├── documents.json
│   └── eval_pairs.json
└── output/
    ├── results.json
    └── retrieval_benchmark.md
```

### Running Tests
```bash
pytest
pytest tests/test_embeddings.py
pytest --cov=src
```

### Example Output
| Metric | Local | GCP (Mock) | Difference |
|--------|-------|-----------|-----------|
| Precision 5 | 0.28 | 0.016 | 0.264 |
| Recall 5 | 1 | 0.06 | 0.94 |
| Mrr | 0.85 | 0.036 | 0.818 |
| Ndcg 5 | 0.90 | 0.036 | 0.860 |

---

## Shoppers Store Upsell

**Repository:** https://github.com/badalsharma9929/Shoppers_store_upsell

### Overview
Shopper Club — Post-Purchase Upsell & Cross-Sell App

A Shopify-inspired Thank You Page upsell engine built for Indian D2C brands. Increases AOV by auto-matching crash-price offers to customers right after checkout — when buying intent is at its peak.

### Live Demo
| Page | URL | What to Try |
|------|-----|-------------|
| 🛍️ **Storefront** | https://gratitude-convert.preview.emergentagent.com/ | Browse 7 spiritual & wellness products |
| 📦 **Sample Product** | https://gratitude-convert.preview.emergentagent.com/product/prod-ganesha-dome | Amazon-style PDP with gallery |
| 👔 **Admin Dashboard** | https://gratitude-convert.preview.emergentagent.com/admin | Manage offers + view analytics |

### Core Concept
Customer buys a product → Thank You Page → Auto-matched crash-price offer → Coupon with 10-min timer → Sticky timer footer across pages

### Tech Stack
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend Framework** | React 19 | UI rendering with hooks-based components |
| **Routing** | React Router v6 | SPA navigation across pages |
| **Styling** | TailwindCSS 3 + CSS Variables | Utility-first responsive styling |
| **UI Components** | Shadcn/UI + Radix | Accessible primitives |
| **State Management** | React Context API | Cart state via `CartContext` |
| **Backend Framework** | FastAPI 0.110 (Python 3.11) | Async REST API |
| **Database** | MongoDB 7 (Motor async driver) | Document store |
| **Fonts** | Outfit (headings) + Manrope (body) | Distinctive Indian D2C aesthetic |

### Features
#### Customer-Facing Storefront
- 🛒 Product Listing with category filters
- 🔍 Amazon-style Product Detail Page
- 🛍️ Smart Cart with localStorage persistence
- 💳 Mock Checkout with COD/Card options
- 🎉 Order Confirmation

#### Post-Purchase Upsell Engine
- 🤖 Auto-Match Algorithm based on product collection/tags
- 🚫 Exclusion Logic (never suggest same product)
- 💥 Crash Price Display
- ⏰ 10-Minute Countdown Timer
- 🎟️ Auto-Generated Coupon Codes
- 🔁 One-Click Redirect with discount auto-applied

#### Sticky Timer Footer
- 📌 Follows customer across all pages
- 🔥 Urgency escalation (red + pulse in last 2 minutes)
- ❌ Dismissible with auto-cleanup
- 🧠 Smart hiding on specific pages

#### Admin Dashboard
- 📊 Stats Cards (total offers, active offers, clicks, CR)
- ➕ Offer Builder (collection-based OR tag-based)
- 🎨 Custom Discount Configuration
- 👁️ Live Preview of each offer
- 🔄 Toggle/Delete offers

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cat > .env <<EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=shopper_club
CORS_ORIGINS=*
EOF
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup
```bash
cd frontend
yarn install
echo "REACT_APP_BACKEND_URL=http://localhost:8001" > .env
yarn start
```

### API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/products` | List all products |
| `POST` | `/api/orders` | Place a new order |
| `GET` | `/api/offers` | List all offers |
| `POST` | `/api/offers/match` | Find best offer for purchased products |
| `POST` | `/api/discount/generate` | Generate unique coupon |
| `GET` | `/api/stats` | Dashboard analytics |

---

## SpendWise

**Repository:** https://github.com/badalsharma9929/spendwise

### Overview
SpendWise - Smart Expense Tracker

A privacy-first expense tracking app for India with automatic transaction capture from UPI apps, bank notifications, and receipt scanning.

### The Problem
Every month, people struggle to track where their money goes. Transactions are scattered across UPI apps (Google Pay, PhonePe, PayTM), credit/debit cards, digital wallets, and subscriptions.

### Solution Architecture
```
Transaction Happens
        ↓
   ┌────┴─────┐
   │          │
Notification  Photo Capture
Listener     (Receipt OCR)
   │          │
   └────┬─────┘
        ↓
   Data Fusion Engine
   (Combine sources)
        ↓
   Auto-Categorization
        ↓
   User Confirmation
        ↓
   Store & Learn
```

### Features
- 📱 **Automatic Transaction Capture** - Captures transactions from bank apps and UPI notifications
- 🧾 **Receipt Scanning** - Scan receipts with OCR to auto-extract amounts
- 🏷️ **Smart Categorization** - Automatically categorizes expenses
- 📊 **Visual Insights** - Beautiful charts showing spending by category
- 🔒 **Privacy First** - All data stored locally on your device
- 🌙 **Dark Mode** - Easy on the eyes theme support

### Supported Apps

#### UPI Apps
- Google Pay
- PhonePe
- PayTM
- Amazon Pay
- BHIM UPI

#### Banks
- SBI, HDFC, ICICI, Axis, Kotak, IDFC, and more

### Tech Stack
| Layer | Technology | Why |
|-------|------------|-----|
| Framework | Flutter 3.24 | Fast, cross-platform |
| Language | Dart | Flutter's native language |
| State | Riverpod | Type-safe, testable |
| Database | SQLite | Reliable, local-only |
| OCR | Google ML Kit | Free, on-device |
| Charts | fl_chart | Beautiful visualizations |
| Architecture | Clean Architecture | Scalable, maintainable |

### Setup
```bash
git clone https://github.com/badalsharma9929/spendwise.git
cd spendwise
flutter pub get
flutter run
```

### Building APK
```bash
flutter build apk --debug
flutter build apk --release
```

### Project Structure
```
lib/
├── core/              # Constants, theme, utilities
├── data/              # Database, models, repositories, services
├── domain/            # Entities, repository interfaces
└── presentation/      # Pages, widgets, providers, router
```

---

## Thank You Upsell AI

**Repository:** https://github.com/badalsharma9929/thank-you-upsell-ai

### Overview
THANKYOUBOOST - An AI-assisted Shopify upsell and bundle automation prototype for product pages, Thank You page offers, and merchant-side revenue analysis.

### Live Demo
https://files-mentioned-by-the-user-shopify-topaz.vercel.app

**Product Page Example:** https://files-mentioned-by-the-user-shopify-topaz.vercel.app/products/luxury-temple-dome

### Core Features
- Merchant dashboard with revenue, bundle checkout, profit, and conversion metrics
- Campaign builder for product, collection, tag, and order-value triggers
- Product catalog with realistic home decor products
- Product detail pages with galleries, price, ratings, stock, category, features
- Live bundle builder with instant subtotal, discount, and final amount updates
- Recommended product section based on shared tags, collections, price fit
- Cart-style checkout panel
- Post-purchase Thank You offer flow with discount links
- AI-style campaign suggestion generator
- Vercel deployment with serverless routing
- Local JSON data persistence

### Products Included
- Temple Significance Dome
- Luxury Temple Dome
- Prosperity Temple Dome
- Travel Blessing Mini Dome
- Mala and Aasan Ritual Set
- Temple Fragrance Set
- Lotus Glow Table Lamp

### Tech Stack
| Area | Technology |
| --- | --- |
| Runtime | Node.js 20+ |
| Server | Native Node HTTP server |
| Frontend | HTML, CSS, vanilla JavaScript |
| State and business logic | Local JSON database plus rule-engine module |
| Recommendation logic | Tag, collection, price, inventory, rating, margin, and add-on scoring |
| Deployment | Vercel serverless functions |

### Architecture
```
Browser
   |
   |-- public/index.html
   |-- public/app.js
   |-- public/styles.css
   |
Node/Vercel handler
   |
   |-- server.js
   |-- api/index.js
   |
Business logic
   |
   |-- src/rule-engine.js
   |
Prototype data
   |
   |-- data/db.json (local only)
```

### API Surface
| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/state` | Returns shop data, products, campaigns, events, and analytics |
| GET | `/api/analytics` | Returns summarized analytics only |
| POST | `/api/campaigns` | Creates a campaign |
| PUT | `/api/campaigns/:id` | Updates a campaign |
| POST | `/api/campaigns/:id/toggle` | Activates or pauses a campaign |
| DELETE | `/api/campaigns/:id` | Deletes a campaign |
| POST | `/api/offer` | Chooses best post-purchase offer |
| POST | `/api/events` | Tracks view, click, purchase, bundle checkout events |
| POST | `/api/ai/suggest-campaign` | Generates campaign recommendation |
| POST | `/api/reset-demo` | Resets prototype data |

### Local Setup
```bash
npm start
# Open: http://localhost:4173
```

Or run directly:
```bash
node server.js
```

### Tests
```bash
npm test
node test/rule-engine.test.js
```

### Deployment
```bash
npx vercel deploy --prod --yes
```

---

## Hatchways - Filtering API

**Repository:** https://github.com/hatchways-community/implementing-a-filtering-and-sorting-api-route-f00e6514cb394eda93ebc1da84e81b37

### Overview
A Flask API implementation with filtering and sorting capabilities.

### System Requirements
- Python v3.11

### Setup

#### Virtual Environment
Set up a virtual environment as per [Flask documentation](https://flask.palletsprojects.com/en/1.1.x/installation/#virtual-environments).

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Seed Database
```bash
python seed.py
```

#### Run Dev Server
```bash
flask run --port=8080
```

### Docker Setup
```bash
docker-compose up
```

To shut down:
```bash
Ctrl-C  # or in separate terminal: docker-compose down
```

### API Testing

#### Example Login (cURL)
```bash
curl --location --request POST 'localhost:8080/api/login' \
--header 'Content-Type: application/json' \
--data-raw '{
    "username": "thomas",
    "password": "123456"
}'
```

#### Create Blog Post (Authenticated)
```bash
curl --location --request POST 'localhost:8080/api/posts' \
--header 'x-access-token: your-token-here' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "This is some text for the blog post...",
    "tags": ["travel", "hotel"]
}'
```

### Helpful Commands
- `black .` - Run auto-formatter
- `flask test` - Run unit tests (non-comprehensive)
- `python seed.py` - Wipe existing data and populate with samples

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Repositories | 13 |
| Public Repositories | 11 |
| Private Repositories | 1 |
| README Files Retrieved | 12 |
| Repositories without README | 2 |

## Technology Summary

### Most Used Languages
1. **JavaScript/TypeScript** - Used in 7+ projects
2. **Python** - Used in 6+ projects
3. **Dart** - Used in 1 project

### Popular Frameworks
- **React/Next.js** - 4 projects
- **FastAPI** - 2 projects
- **Streamlit** - 3 projects
- **Flask** - 1 project
- **Flutter** - 1 project

### Key Technologies
- **AI/ML**: LLMs, RAG, Vector Search, NLP, Transformers
- **Databases**: MongoDB, PostgreSQL, SQLite, FAISS
- **Deployment**: Vercel, Streamlit Cloud, Railway, Docker
- **Frontend**: React, TypeScript, TailwindCSS, Vue.js
- **Backend**: Node.js, Python, Express, FastAPI, NestJS

---

**Document Created:** June 2026  
**Author:** Badal Sharma (@badalsharma9929)  
**Contact:** [GitHub Profile](https://github.com/badalsharma9929)
