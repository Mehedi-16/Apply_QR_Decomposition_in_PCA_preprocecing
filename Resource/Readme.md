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

### ১. পরিচিতি (Introduction)

**Principal Component Analysis (PCA)** হলো একটি জনপ্রিয় ও গুরুত্বপূর্ণ পদ্ধতি, যার মাধ্যমে আমরা ডেটার মাত্রা (dimensionality) কমিয়ে দিই। এতে করে ডেটা বিশ্লেষণ করা সহজ হয়। PCA ব্যবহার করা হয় যেমন pattern recognition বা data representation এর কাজে।

---

### কীভাবে কাজ করে PCA?

ধরা যাক তোমার কাছে একটা training data সেট আছে, যেটার মধ্যে 𝑛 টি নমুনা (samples) আছে, আর প্রতিটি নমুনা 𝑑 মাত্রার (dimensions)। অর্থাৎ, ডেটা দেখতে এমন —
$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n] \in \mathbb{R}^{d \times n}$
এখানে, 𝑑 হলো ডেটার মাত্রা, আর 𝑛 হলো নমুনার সংখ্যা।

PCA-তে আমরা ক্লাস লেবেল (যেমন কোন নমুনা কোন গ্রুপের) ব্যবহার করি না, অর্থাৎ এটি non-supervised learning।

---

### ডেটা থেকে কী বের করি?

প্রথমে আমরা ডেটার **covariance matrix (Σₓ)** তৈরি করি, যা ডেটার মধ্যে সম্পর্ক বোঝায়।
এরপর এই covariance matrix-এর **Eigenvalue Decomposition (EVD)** করি, যার মাধ্যমে বড় বড় eigenvalue ও তাদের সাথে যুক্ত eigenvector গুলো পাই।

**উদাহরণ:**
ধরা যাক তোমার covariance matrix 3×3 এর এবং eigenvalues হলো 5, 2, 1। তাহলে PCA পদ্ধতিতে তুমি সবচেয়ে বড় eigenvalue (5) এর eigenvector থেকে শুরু করে দরকারি eigenvectorগুলো বেছে নেবে। এর মাধ্যমে তুমি ডেটার মাত্রা কমিয়ে ছোট একটি স্পেস তৈরি করবে, যেখানে ডেটার সবচেয়ে গুরুত্বপূর্ণ তথ্য থাকবে।

---

### সমস্যা: ছোট নমুনার সংখ্যা (Small Sample Size Problem)

অনেক সময় ডেটার মাত্রা (𝑑) অনেক বেশি কিন্তু training sample সংখ্যা (𝑛) কম হয়। যেমন face recognition বা biometrics-এ হয়।
এই সমস্যাকে বলা হয় **“small sample size problem”**।

এই ক্ষেত্রে covariance matrix 𝑑×𝑑 এর বড় একটা matrix হয়। তার EVD বের করা অনেক সময়সাপেক্ষ এবং কম্পিউটারের জন্য ভারি কাজ।

---

### কীভাবে সমস্যা সমাধান হয়?

Fukunaga নামের একজন গবেষক বলেছেন, covariance matrix এর পরিবর্তে 𝐇ᵀ𝐇 এর EVD করা যেতে পারে, যা তুলনামূলক দ্রুত।

তবে এই পদ্ধতিটা একটা সমস্যা আছে — এটি **numerically unstable** হতে পারে। অর্থাৎ, কম্পিউটারের গণনায় ছোট ছোট সংখ্যার কারণে ভুল হতে পারে এবং তথ্য হারাতে পারে।

---

### আরেকটি পদ্ধতি: SVD

এই সমস্যার কারণে, সরাসরি covariance matrix না নিয়ে, আমরা **Singular Value Decomposition (SVD)** ব্যবহার করি **𝐇** ম্যাট্রিক্সের উপর।

কিন্তু SVD করতেও অনেক গণনা লাগে — প্রায় **14𝑑𝑛² flops**।

---

### আরো দ্রুত কিন্তু সীমিত সঠিকতা পদ্ধতি

কিছু পদ্ধতি আছে যেগুলো PCA দ্রুত করার চেষ্টা করে কিন্তু সেগুলোর সঠিকতা কম। যেমন **Postponed Basis Matrix Multiplication (PBM)-PCA** যেখানে ডেটা matrix কে তিনটি matrix-এ ভাগ করে কাজ করা হয়।

এটা একটু approximation, অর্থাৎ সঠিক তথ্য পুরোপুরি পাওয়া যায় না।

---

### এই গবেষণার উদ্দেশ্য

আমাদের লক্ষ্য হলো এমন একটা PCA পদ্ধতি তৈরি করা, যা —

* **দ্রুত** (computationally efficient)
* এবং **সঠিক** (numerically stable) হবে।

আমরা stability বাড়ানোর জন্য matrix 𝐇 ব্যবহার করেছি, আর efficiency বাড়ানোর জন্য **QR decomposition** ব্যবহার করেছি।

এই পদ্ধতির জন্য লাগে মাত্র প্রায় **2𝑑𝑛² + 2𝑑ℎ² flops**, যা SVD পদ্ধতির তুলনায় অনেক কম।

---

### ছোট্ট সারসংক্ষেপ উদাহরণ

* ধরা যাক, তোমার ডেটার মাত্রা (𝑑) = 100 এবং নমুনার সংখ্যা (𝑛) = 50।
* SVD পদ্ধতিতে প্রয়োজন হতে পারে প্রায় ১৪ × ১০০ × ৫০² = ৩৫,০০,০০০ flops।
* আমাদের QR ভিত্তিক পদ্ধতিতে লাগবে প্রায় ২ × ১০০ × ৫০² + ২ × ১০০ × ℎ² flops, যেখানে ℎ (কম করা মাত্রা) ধরা যাক ২০।
* তাহলে, ২ × ১০০ × ২৫০০ + ২ × ১০০ × ৪০০ = ৫,০০,০০০ + ৮০,০০০ = ৫,৮০,০০০ flops।
* অর্থাৎ, QR পদ্ধতিতে কম্পিউটেশন অনেক কম হয় এবং কাজ দ্রুত হয়।

---

### তোমার সুবিধার জন্য মূল বিষয়গুলো সংক্ষেপে:

* PCA ডেটার মাত্রা কমানোর পদ্ধতি।
* Covariance matrix এর eigenvalue decomposition করতে সময় লাগে বেশি।
* SVD দ্রুত কিন্তু বেশি গণনা করে।
* QR decomposition ব্যবহার করে PCA করলে কম সময় লাগে এবং সঠিকতাও থাকে।

---

