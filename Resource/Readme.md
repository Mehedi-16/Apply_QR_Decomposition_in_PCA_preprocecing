**Principal Component Analysis using QR Decomposition**
**লেখক:** Alok Sharma, Kuldip K. Paliwal, Seiya Imoto, Satoru Miyano

**Abstract:**
In this paper, **আমরা প্রস্তাব করেছি** একটা নতুন PCA method যেটা **QR বিযুক্তি (QR Decomposition)** এর ওপর ভিত্তি করে।
এই পদ্ধতিটা **সংখ্যাগতভাবে স্থিতিশীল (Numerically stable)**, ঠিক যেমনটা **SVD ভিত্তিক PCA** হয়।

আমরা দু’ধরনের তুলনা করেছি —

1. **বিশ্লেষণাত্মক তুলনা (Analytical comparison)**
2. **গাণিতিক সিমুলেশন** using MATLAB software

আমাদের মূল উদ্দেশ্য ছিল এই method-এর **গণনাগত জটিলতা (Computational complexity)** পরীক্ষা করা।

---

✅ **SVD ভিত্তিক PCA** requires almost **14𝑑𝑛² flops**,
where:
– 𝑑 = **ফিচার স্পেসের মাত্রা**
– 𝑛 = **প্রশিক্ষণ ডেটার সংখ্যা**

✅ আর **QR ভিত্তিক PCA** তে লাগে প্রায় **2𝑑𝑛² + 2𝑑𝑡ℎ flops**,
where:
– 𝑡 = **ডেটা কোভেরিয়েন্স ম্যাট্রিক্সের র‍্যাঙ্ক**
– ℎ = **সংক্ষিপ্ত ফিচার স্পেসের মাত্রা**

---

🔍 So, **আমরা দেখতে পাই** QR based PCA is more **efficient**,
because it takes **less computation** and works faster in many cases.

---
---

### **মূল শব্দসমূহ (Keywords):**

Principal Component Analysis;
Singular Value Decomposition;
QR Decomposition;
Computational Complexity.

চাইলে এগুলোর ছোট ছোট ব্যাখ্যা বাংলায় আলাদাভাবে লিখে দিতে পারি, যেমন:

* **Principal Component Analysis (PCA):** ডেটা থেকে গুরুত্বপূর্ণ ফিচার নির্বাচন করার একটি পরিসংখ্যানগত পদ্ধতি।
* **Singular Value Decomposition (SVD):** একটি ম্যাট্রিক্স ভাঙার (decompose) পদ্ধতি যা PCA-তে ব্যবহৃত হয়।
* **QR Decomposition:** একটি ম্যাট্রিক্সকে দুইটি উপাদানে ভাঙার পদ্ধতি—একটি orthogonal (Q) এবং একটি upper triangular (R)।
* **Computational Complexity:** একটি অ্যালগরিদম কতটা সময় ও রিসোর্স ব্যবহার করে তা পরিমাপ করার উপায়।
---
---
