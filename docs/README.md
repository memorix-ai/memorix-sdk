# 📚 Memorix SDK Documentation

Welcome to the Memorix SDK documentation! This directory contains comprehensive guides and references for using the Memorix SDK.

## 📖 Getting Started

- **[Installation Guide](INSTALL.md)** - Complete setup instructions for all platforms
- **[Quick Reference](QUICK_REFERENCE.md)** - Essential commands and patterns at a glance
- **[Usage Guide](USAGE.md)** - Comprehensive usage examples and best practices

## 🏗️ Architecture & Design

- **[Architecture](ARCHITECTURE.md)** - System design, components, and data flow
- **[Vision](VISION.md)** - Project vision, mission, and roadmap

## 🛠️ Development

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](CHANGELOG.md)** - Version history and release notes

## 📋 Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [Installation](INSTALL.md) | Setup and configuration | New users |
| [Quick Reference](QUICK_REFERENCE.md) | Essential commands | All users |
| [Usage Guide](USAGE.md) | Advanced usage patterns | Developers |
| [Architecture](ARCHITECTURE.md) | System design | Architects |
| [Vision](VISION.md) | Project direction | Stakeholders |
| [Contributing](CONTRIBUTING.md) | Development guidelines | Contributors |
| [Changelog](CHANGELOG.md) | Version history | All users |

## 🚀 Quick Start

```python
from memorix import MemoryAPI, Config

# Initialize
config = Config('memorix.yaml')
memory = MemoryAPI(config)

# Store and retrieve
memory_id = memory.store("Hello, Memorix!")
results = memory.retrieve("Hello")
```

## 📞 Support

- **Documentation**: [docs.memorix.ai](https://docs.memorix.ai)
- **Issues**: [GitHub Issues](https://github.com/memorix-ai/memorix-sdk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/memorix-ai/memorix-sdk/discussions)
- **Email**: support@memorix.ai

---

*For the main project README, see [../README.md](../README.md)* 