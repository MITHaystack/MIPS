$PROJECT = 'MIPS'
$ACTIVITIES = [
              'version_bump',  # Changes the version number in various source files (setup.py, __init__.py, etc)
              'changelog',  # Uses files in the news folder to create a changelog for release
              'tag',  # Creates a tag for the new version number
               ]

$CHANGELOG_FILENAME = 'CHANGELOG.rst'  # Filename for the changelog
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'  # Filename for the news template
$PUSH_TAG_REMOTE = 'git@github.com:MITHaystack/MIPS.git'  # Repo to push tags to
$WEBSITE_URL = "https://github.com/MITHaystack/MIPS"
$GITHUB_ORG = 'MITHaystack'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'MIPS'  # Github repo for Github releases  and conda-forge
