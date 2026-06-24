# 🧠 Neuroeduai — Miguel Fernandes

> **Personal website of Miguel Fernandes — Senior Data Scientist.**
> Highlighting the intersection of healthcare, neuroscience, education, genetics, and artificial intelligence.

[![Built with Hugo](https://img.shields.io/badge/Hugo-v0.148.1-brightgreen.svg?style=flat-square&logo=gohugo)](https://gohugo.io/)
[![Framework: Hugo Blox](https://img.shields.io/badge/Framework-Hugo%20Blox-blue.svg?style=flat-square)](https://hugoblox.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE.md)

---

## 🌐 Overview

This is the source code repository for [Neuroeduai](https://www.neuroeduai.com/), the personal website and academic portfolio of **Miguel Fernandes**.

The website is designed to share insights and research across:
*   🏥 **Healthcare & Medical Treatment Optimization**
*   🧠 **Neuroscience & Cognitive Science**
*   🎓 **Education & Knowledge Empowerment**
*   📊 **Data Science (AI / Machine Learning / Deep Learning)**
*   🧬 **Genetics & Life Sciences**

---

## 🛠️ Built With

*   **Static Site Generator:** [Hugo](https://gohugo.io/) (Extended Edition)
*   **Engine & Theme:** [Hugo Blox Builder](https://hugoblox.com/) (formerly *Wowchemy*) — a modular, block-based website builder.
*   **Hosting & Deployment:** Automated builds on [Netlify](https://www.netlify.com/).

---

## 🚀 Local Development

Follow these steps to preview and build the website locally.

### 📋 Prerequisites

Ensure you have the **Hugo Extended Edition** installed. You can install it on macOS using Homebrew:

```bash
brew install hugo
```

To verify your installation:

```bash
hugo version
```

### 💻 Running the Development Server

Start the Hugo development server with drafts and future-dated posts enabled:

```bash
hugo server -D
```

By default, the server will watch for changes and rebuild the site automatically. 
Open [http://localhost:1313](http://localhost:1313) in your browser to view the site.

---

## 📦 Managing Dependencies (Hugo Blox Builder)

The project leverages Hugo modules for the underlying engine. To keep the framework modules up-to-date:

### Update to the Latest Version

```bash
hugo mod get -u
```

### Clean Up Module Cache

```bash
hugo mod tidy
```



## ✈️ Deployment

This website is configured for continuous integration and delivery.
*   Every push or pull request to the `main` branch triggers a build on **Netlify**.
*   Configuration options (such as Hugo versions) are specified in the [`netlify.toml`] file.

---

## 📄 License

Code is licensed under the [MIT License](LICENSE.md).
Content and publications are copyright to their respective authors/publishers.
