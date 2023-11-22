import click


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("g_max", type=float)
def generate(g_max) -> None:
    pass
