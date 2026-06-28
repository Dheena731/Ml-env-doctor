(() => {
  const data = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "mlenvdoctor",
    applicationCategory: "DeveloperApplication",
    operatingSystem: "Windows, macOS, Linux",
    description: "ML environment diagnostics for PyTorch, CUDA, Conda, Pipenv, Docker, and LLM training stacks.",
    softwareVersion: "0.1.6",
    offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    codeRepository: "https://github.com/Dheena731/Ml-env-doctor"
  };
  const script = document.createElement("script");
  script.type = "application/ld+json";
  script.text = JSON.stringify(data);
  document.head.appendChild(script);
})();
