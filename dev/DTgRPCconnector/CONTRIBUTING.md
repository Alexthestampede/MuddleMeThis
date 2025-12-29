# Contributing to Draw Things gRPC Python Client

Thank you for your interest in contributing! This project welcomes contributions from the community.

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Provide clear description and steps to reproduce
- Include Python version, OS, and server setup details
- Add relevant error messages and logs

### Submitting Changes

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test thoroughly**
   ```bash
   # Test your changes
   python examples/generate_image.py "test prompt"
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone repository
git clone <repository-url>
cd gRPC

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_qwen_lora.py
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small

## Adding Examples

If you have useful patterns or use cases:

1. Add to `EXAMPLES.md`
2. Include clear description
3. Show expected output
4. Mention any gotchas

## Testing

- Test with multiple models when possible
- Verify both compressed and uncompressed responses
- Check edge cases (wrong dimensions, invalid models, etc.)

## Documentation

- Update README.md for major features
- Add examples for new functionality
- Document any new parameters or options
- Keep EXAMPLES.md comprehensive

## Questions?

Feel free to open an issue for:
- Feature requests
- Implementation questions
- Documentation clarifications
- General discussion

Thank you for contributing!
