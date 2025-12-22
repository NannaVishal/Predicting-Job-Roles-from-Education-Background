# users/management/commands/train_model.py
from django.core.management.base import BaseCommand, CommandError
from users import ml_helper

class Command(BaseCommand):
    help = 'Train job role model from a file (csv or xlsx).'

    def add_arguments(self, parser):
        parser.add_argument('--path', required=True, help='Path to CSV or XLSX file')
        parser.add_argument('--label_col', default='job_role', help='Name of target label column')

    def handle(self, *args, **options):
        path = options['path']
        label_col = options['label_col']
        try:
            model_path = ml_helper.train_model_from_file(path, label_col=label_col)
            self.stdout.write(self.style.SUCCESS(f'Model trained and saved to {model_path}'))
        except Exception as e:
            raise CommandError(str(e))
